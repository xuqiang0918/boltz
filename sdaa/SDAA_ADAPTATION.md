# Boltz2 SDAA 适配说明

> **结论**：Boltz2 在 SDAA 上已跑通推理（结构预测 + 结合亲和力），与 CUDA（A100-40GB）的精度  
> 差异**处于扩散采样固有随机性的量级之内** —— 11 个真实 MSA 测例的 `confidence_score` 平均偏差  
> 3.05e-04、最大 8.89e-04，结构未叠合 RMSD 0.020 ~ 0.264 Å，两项都**小于** CUDA 自己换一个 seed  
> 造成的波动；① SDAA 出厂状态未叠合 RMSD 达 17.6 ~ 88.9 Å。完成方式为 1 处加速器适配 + 1 处  
> 上游兼容修复 + 1 处算子缺陷规避（第 2 节），外加运行前设一个平台环境变量对齐设备随机数  
> （第 3 节），不改模型逻辑。  
> **三组结果（① SDAA 原始 / ② CUDA / ③ SDAA 对齐）见第 5 章。**  
> 性能：11 个测试蛋白（56 ~ 363 aa）单卡单例下 SDAA / CUDA 预测耗时 **3.9× ~ 34.8×**（合计 18.4×），见第 6 章。

## 1. 适配环境

| 项目      | 说明                                           |
| --------- | ---------------------------------------------- |
| 模型      | Boltz2（仓库 tag v2.2.1）                      |
| 加速平台  | SDAA（Torch-SDAA 3.2.0，基于 PyTorch 2.7.1）   |
| CUDA 基线 | NVIDIA A100-40GB（仅作对比参考，见第 5、6 章） |
| Python    | 3.11                                           |

关键平台事实：Teco 构建的 PyTorch 中 `torch.cuda` 存在但 `is_available()` 返回 False，真实加速器由 `torch.sdaa` 暴露；Lightning 默认无法识别 sdaa 设备，需注册自定义 Accelerator。

## 2. 适配改动

新增 `sdaa/sdaa_acc.py`：`SDAAAccelerator` 继承 Lightning 的 `CUDAAccelerator`，覆写 6 处方法  
（`setup_device` / `parse_devices` / `get_parallel_devices` / `auto_device_count` / `is_available` /  
`register_accelerators`），使 Lightning connector 的 root_device 走 CUDA 分支逻辑。前 5 处的父类实现  
依赖 `torch.cuda` 与 `num_cuda_devices()`，在 SDAA 上会返回 0 张卡或直接 `raise`（实测 `parse_devices(1)`  
报 "your machine only has: []"、`auto_device_count()` 返回 0、`setup_device` 报 "Device should be GPU"）；  
`register_accelerators` 的父类实现用 `cls.name()` 注册，而 `name()` 经注册表反查会得到 `"cuda"`，  
会把 CUDA 加速器顶掉。`setup` / `teardown` / `get_device_stats` 的父类实现本平台实测可正常执行  
（`set_nvidia_flags(0)`、`_clear_cuda_memory()` 均不报错），故不再覆写。

修改 `src/boltz/main.py`：

- `[sdaa-adapt]` `--accelerator` 增加 `sdaa` 选项；选择 sdaa 时将 `SDAAAccelerator` 注册进 `AcceleratorRegistry` 并向 Trainer 传入其**实例**（传实例才能通过 connector 对 CUDA 加速器的 isinstance 白名单检查，root_device 才会正确落到 sdaa 设备）。
- `[upstream]` 两处 `torch.load(..., weights_only=False)`：PyTorch 2.7 下加载官方 checkpoint 需要显式关闭 weights_only，否则报错。

`[sdaa-adapt]` 修改 `src/boltz/model/modules/` 下 4 个文件，规避 SDAA 的  
`nn.Embedding` rank≥3 缺陷（复现脚本 `sdaa/repro_embedding_sdaa.py`）：

- 新增 `embedding_lookup(embedding, indices)`（`model/modules/utils.py`）：仅当索引张量位于  
  sdaa 设备且 `dim() >= 3` 时改走 `torch.index_select`；rank≤2、非 sdaa 设备、以及带  
  `padding_idx` / `max_norm` 的层一律保持原始调用。
- 替换 5 处调用点：

| 文件                            | 行  | 查表层                             | 索引形状  |
| ------------------------------- | --- | ---------------------------------- | --------- |
| `model/models/boltz2.py`        | 428 | `token_bonds_type` (7,128)         | `(1,N,N)` |
| `model/modules/confidencev2.py` | 170 | `token_bonds_type` (7,128)         | `(1,N,N)` |
| `model/modules/confidencev2.py` | 199 | `dist_bin_pairwise_embed` (64,128) | `(1,N,N)` |
| `model/modules/confidence.py`   | 294 | `dist_bin_pairwise_embed` (64,128) | `(1,N,N)` |
| `model/modules/affinity.py`     | 108 | `dist_bin_pairwise_embed` (64,128) | `(1,N,N)` |

除上述两处外，适配不修改其他模型/数据代码；rank≤2 的查表与非 SDAA 设备上的行为与官方完全一致。

设备随机数与 CUDA 的对齐**不在代码里**，是运行前设置一个平台环境变量（见第 3 节），所以不产生额外的代码改动。

## 3. 运行前准备

1. **设置设备随机数对齐（必做）**：运行前先 `export TORCH_SDAA_ALIGN_NV_DEVICE=a100`。SDAA 出厂的设备 generator（TecoRAND）与 CUDA（Philox）不是同一条序列，不设会让 50 步扩散偏到另一个结构、与 CUDA 不可比（即第 5 章的 ①）。该变量由平台库 `libtorch_sdaa.so` 解析（取值 `a100` / `v100`），只要在**第一次消费设备随机数之前**设好即可，启动前 export 足够。⚠️ 不要设成 `0` / `off` / `none`：它按设备名解析，会直接 abort；要复现未对齐的 ① 就干脆不设这个变量。
2. 按仓库 requirements 安装依赖（rdkit、biopython、gemmi 等），并安装 Torch-SDAA 环境。
3. 模型权重与 CCD 缓存：首次运行 `boltz predict` 会自动下载到 `~/.boltz`（约 7GB）；离线环境可手动准备目录并通过 `--cache` 指定，结构如下：

```
<cache_dir>/
├── boltz2_conf.ckpt    # 结构预测权重
├── boltz2_aff.ckpt     # 亲和力预测权重
└── mols/               # CCD 分子缓存
```

## 4. 用户自测自己的数据

### 4.1 输入格式

每个任务一个 yaml 文件（把多个 yaml 放进同一目录即可批量推理；`boltz predict` 的 DATA 参数只收单个路径，多输入需传目录）。yaml 描述复合物组成：

```yaml
version: 1
sequences:
- protein:
    id: A
    sequence: <蛋白序列>
    msa: <a3m文件路径>          # protein 链必须提供 msa 字段
- ligand:
    id: L
    smiles: <SMILES>            # 或用 ccd: <CCD码>，两者二选一
```

支持的实体：`protein` / `dna` / `rna`（序列输入）、`ligand`（SMILES 或 CCD 码）、修饰残基（`modifications`，仅 CCD）、环肽（`cyclic`）。离子按 ligand + CCD 码处理。

MSA 说明：protein 链的 `msa` 字段指向 a3m 文件；没有现成 MSA 时可用"单序列 a3m"（仅含 query 一条，精度会有损失），格式就两行：

```
>query_name
<与 sequence 完全一致的序列>
```

注意：复合物中多条蛋白链序列相同时，必须共享同一个 a3m 文件路径，否则报错 "All proteins with same sequence must share same MSA"。

### 4.2 官方测例（examples/）

仓库 `examples/` 自带官方测例，可直接用作验证输入。离线（无 MSA server）环境注意：

- 官方 yaml 中**不带 `msa` 字段**的蛋白链（如 `prot.yaml` / `cyclic_prot.yaml`）会报  
  `Missing MSA's in input and --use_msa_server flag not set` 被跳过——这是官方行为，需要联网 MSA server；
- 离线可跑的官方写法：`prot_no_msa.yaml`（`msa: empty`，单序列无 MSA）、`prot_custom_msa.yaml`（本地 a3m）；
- fasta 输入（`prot.fasta` / `ligand.fasta`）可直接作为 predict 输入；
- `affinity.yaml`（337aa 蛋白 + TYR 配体 + affinity properties）无需 MSA，可离线跑。

### 4.3 结构预测

```bash
export TORCH_SDAA_ALIGN_NV_DEVICE=a100      # 设备随机数对齐，见第 3 节
boltz predict <yaml路径或目录> \
  --accelerator sdaa --devices 1 \
  --model boltz2 \
  --cache <cache_dir> \
  --out_dir <输出目录> \
  --seed 42
```

可选参数：`--sampling_steps`（默认 200）、`--output_format pdb`（默认 mmcif）。批量模式下已产出结果的用例自动跳过，重跑加 `--override`。

### 4.4 结合亲和力预测

在 yaml 末尾追加 properties 块即可，命令不变：

```yaml
properties:
- affinity:
    binder: L     # 指定配体的链 id
```

执行时先跑结构预测，再自动进入亲和力分支（默认 200 步 × 5 个扩散样本），输出额外的 affinity json。

限制：一次只能指定一个小分子配体，重原子数 ≤128（建议 ≤56）；仅对小分子–蛋白靶点可靠。

### 4.5 输出解读

```
<输出目录>/boltz_results_<输入名>/predictions/<用例名>/
├── <用例名>_model_0.cif          # 预测结构（B 因子列 = pLDDT）
├── confidence_<用例名>_model_0.json
├── plddt_/pae_/pde_<用例名>_model_0.npz
└── affinity_<用例名>.json        # 仅 yaml 含 properties 时输出
```

- confidence json 主要字段：`confidence_score = 0.8 × complex_plddt + 0.2 × iptm`（单链时用 ptm），范围 [0,1] 越高越好；多个扩散样本按该分数排序，`model_0` 最优。
- affinity json 两个字段：
  - `affinity_probability_binary`（0~1）：判定 binder/decoy，用于虚拟筛选（hit discovery）；
  - `affinity_pred_value`：log10(IC50)，IC50 单位 μM，数值越低结合越强，用于分子优化排序（hit-to-lead）。换算 pIC50：`(6 − y) × 1.364` kcal/mol。

## 5. CUDA 与 SDAA 精度对比

### 5.1 三组结果与判据

本节回答一个问题：**SDAA 上的 Boltz2 与 CUDA 上是不是同一个模型。** 对比三组结果：

| 编号 | 配置      | 设备随机数                                                                    |
| ---- | --------- | ----------------------------------------------------------------------------- |
| ①    | SDAA 原始 | 出厂 TecoRAND generator，不做任何处理                                         |
| ②    | CUDA      | Philox（参考基准）                                                            |
| ③    | SDAA 对齐 | 与 ② **同一条序列**（`TORCH_SDAA_ALIGN_NV_DEVICE=a100`，第 3 节的运行前设置） |

三平台的设备随机数算法本就不同（CPU = MT19937、CUDA = Philox、SDAA 出厂 = TecoRAND），  
同 seed 抽出的不是同一条序列。50 步扩散逐级放大后，① 会偏离到另一个结构（见 5.3）；  
③ 让 SDAA 复用 CUDA 的序列，两平台才具备可比性。

三臂除 `--accelerator` 与是否设置 `TORCH_SDAA_ALIGN_NV_DEVICE` 外命令行逐字相同，读同一份输入  
（11 个 yaml + 11 个 a3m，md5 两侧逐位一致）：

| 项       | ② CUDA 臂                                 | ①③ SDAA 臂                       |
| -------- | ----------------------------------------- | -------------------------------- |
| 硬件     | NVIDIA A100-40GB                          | 太初 SDAA（单卡）                |
| 命令行   | `--accelerator gpu`                       | `--accelerator sdaa --devices 1` |
| 采样     | `--recycling_steps 3 --sampling_steps 50` | 同左                             |
| MSA 上限 | `--max_msa_seqs 2048`                     | 同左                             |
| 其他     | `--model boltz2 --no_kernels --seed 42`   | 同左                             |

**判据**：扩散采样本身有随机性，「两平台逐位相同」并不成立，所以判据不是「差为 0」，  
而是 **平台差异 ≤ CUDA 自己换一个 seed 造成的波动**（同机同参数 seed 42 → 7），后者是差异的合理上界。  
复现 ① 就是不设 `TORCH_SDAA_ALIGN_NV_DEVICE`（出厂状态）。

> 注意：三臂必须走官方 CLI。外面套一层探针 wrapper（先 `import torch_sdaa` 并探测显存）会改变随机数的  
> 消费顺序，结果与 CLI 不一致 —— 实测该口径下 CUDA 侧 1pgb 的 `confidence_score` 为 0.950483、  
> SDAA 侧与 ① 逐位相同（0.78689241）；同样 seed 42 下 CLI 复跑两次逐位一致（0.9477965235710144 ×2），  
> 说明差异来自 wrapper 而不是采样本身不稳定。

### 5.2 测例

序列取自 RCSB 真实结构条目，MSA 由 ColabFold 服务（`/ticket/msa`，`mode=env`）生成、  
交给 boltz 自身的 mmseqs2 解析器落盘后**冻结**，三臂读同一份 —— 不是单序列占位 MSA。

| 测例 | 长度   | MSA 条数 | 来源      |
| ---- | ------ | -------- | --------- |
| 1pgb | 56 aa  | 144      | RCSB 1PGB |
| 1csp | 67 aa  | 9518     | RCSB 1CSP |
| 1ubq | 76 aa  | 9655     | RCSB 1UBQ |
| 2ci2 | 83 aa  | 2190     | RCSB 2CI2 |
| 1hhp | 99 aa  | 4953     | RCSB 1HHP |
| 1hel | 129 aa | 3947     | RCSB 1HEL |
| 2lzm | 164 aa | 1180     | RCSB 2LZM |
| 1ake | 214 aa | 7627     | RCSB 1AKE |
| 1gfl | 238 aa | 352      | RCSB 1GFL |
| 1tim | 247 aa | 7470     | RCSB 1TIM |
| 1ald | 363 aa | 4083     | RCSB 1ALD |

覆盖 56 ~ 363 aa 的单链蛋白；MSA 规模从 144 条到 9655 条（`--max_msa_seqs 2048` 截断）。

### 5.3 打分指标（逐测例）

Δ = |SDAA − CUDA|，① 列为未做任何处理的出厂状态，③ 列为交付状态（对齐）。

**confidence_score**（越高越好）

| 测例 | ① SDAA 原始 | ② CUDA | ③ SDAA 对齐 | Δ①       | Δ③           |
| ---- | ----------- | ------ | ----------- | -------- | ------------ |
| 1pgb | 0.9468      | 0.9478 | 0.9483      | 9.70e-04 | **5.24e-04** |
| 1csp | 0.7869      | 0.9505 | 0.9501      | 1.64e-01 | **3.89e-04** |
| 1ubq | 0.7459      | 0.9321 | 0.9319      | 1.86e-01 | **1.86e-04** |
| 2ci2 | 0.7241      | 0.7938 | 0.7933      | 6.97e-02 | **5.28e-04** |
| 1hhp | 0.7583      | 0.8877 | 0.8880      | 1.29e-01 | **2.66e-04** |
| 1hel | 0.8473      | 0.9787 | 0.9785      | 1.31e-01 | **1.60e-04** |
| 2lzm | 0.8712      | 0.9716 | 0.9714      | 1.00e-01 | **2.45e-04** |
| 1ake | 0.9003      | 0.9030 | 0.9031      | 2.73e-03 | **5.04e-05** |
| 1gfl | 0.9441      | 0.9443 | 0.9435      | 2.17e-04 | **8.89e-04** |
| 1tim | 0.9591      | 0.9598 | 0.9598      | 7.35e-04 | **3.15e-05** |
| 1ald | 0.9523      | 0.9519 | 0.9518      | 4.10e-04 | **8.55e-05** |

**ptm**（越高越好）

| 测例 | ① SDAA 原始 | ② CUDA | ③ SDAA 对齐 | Δ①       | Δ③           |
| ---- | ----------- | ------ | ----------- | -------- | ------------ |
| 1pgb | 0.9223      | 0.9223 | 0.9237      | 3.80e-05 | **1.45e-03** |
| 1csp | 0.6408      | 0.9138 | 0.9137      | 2.73e-01 | **1.31e-04** |
| 1ubq | 0.6467      | 0.9122 | 0.9121      | 2.65e-01 | **1.35e-04** |
| 2ci2 | 0.5969      | 0.7410 | 0.7399      | 1.44e-01 | **1.05e-03** |
| 1hhp | 0.7305      | 0.8439 | 0.8451      | 1.13e-01 | **1.12e-03** |
| 1hel | 0.8302      | 0.9609 | 0.9603      | 1.31e-01 | **5.68e-04** |
| 2lzm | 0.8395      | 0.9434 | 0.9418      | 1.04e-01 | **1.59e-03** |
| 1ake | 0.8412      | 0.8528 | 0.8524      | 1.16e-02 | **3.37e-04** |
| 1gfl | 0.9314      | 0.9339 | 0.9329      | 2.48e-03 | **1.01e-03** |
| 1tim | 0.9641      | 0.9644 | 0.9642      | 3.50e-04 | **2.23e-04** |
| 1ald | 0.9506      | 0.9493 | 0.9489      | 1.35e-03 | **4.27e-04** |

**complex_plddt**（越高越好）

| 测例 | ① SDAA 原始 | ② CUDA | ③ SDAA 对齐 | Δ①       | Δ③           |
| ---- | ----------- | ------ | ----------- | -------- | ------------ |
| 1pgb | 0.9530      | 0.9542 | 0.9545      | 1.22e-03 | **2.92e-04** |
| 1csp | 0.8234      | 0.9597 | 0.9592      | 1.36e-01 | **4.53e-04** |
| 1ubq | 0.7706      | 0.9370 | 0.9368      | 1.66e-01 | **1.99e-04** |
| 2ci2 | 0.7559      | 0.8070 | 0.8066      | 5.11e-02 | **3.99e-04** |
| 1hhp | 0.7652      | 0.8987 | 0.8987      | 1.33e-01 | **5.26e-05** |
| 1hel | 0.8516      | 0.9831 | 0.9831      | 1.32e-01 | **5.86e-05** |
| 2lzm | 0.8791      | 0.9787 | 0.9788      | 9.96e-02 | **9.27e-05** |
| 1ake | 0.9151      | 0.9156 | 0.9157      | 5.19e-04 | **1.47e-04** |
| 1gfl | 0.9473      | 0.9470 | 0.9461      | 3.49e-04 | **8.60e-04** |
| 1tim | 0.9578      | 0.9587 | 0.9587      | 8.31e-04 | **1.63e-05** |
| 1ald | 0.9527      | 0.9525 | 0.9525      | 1.77e-04 | **5.96e-08** |

**complex_pde**（越低越好）

| 测例 | ① SDAA 原始 | ② CUDA | ③ SDAA 对齐 | Δ①       | Δ③           |
| ---- | ----------- | ------ | ----------- | -------- | ------------ |
| 1pgb | 0.3341      | 0.3353 | 0.3318      | 1.17e-03 | **3.50e-03** |
| 1csp | 2.5320      | 0.3310 | 0.3319      | 2.20e+00 | **9.17e-04** |
| 1ubq | 1.6465      | 0.3336 | 0.3330      | 1.31e+00 | **6.61e-04** |
| 2ci2 | 1.9443      | 0.6866 | 0.6861      | 1.26e+00 | **5.16e-04** |
| 1hhp | 2.0784      | 0.6116 | 0.6079      | 1.47e+00 | **3.66e-03** |
| 1hel | 0.8102      | 0.2842 | 0.2848      | 5.26e-01 | **6.40e-04** |
| 2lzm | 1.2024      | 0.2839 | 0.2853      | 9.18e-01 | **1.40e-03** |
| 1ake | 0.3707      | 0.3713 | 0.3721      | 6.08e-04 | **8.32e-04** |
| 1gfl | 0.3891      | 0.3785 | 0.3814      | 1.06e-02 | **2.84e-03** |
| 1tim | 0.3407      | 0.3387 | 0.3394      | 2.01e-03 | **7.44e-04** |
| 1ald | 0.4405      | 0.4487 | 0.4520      | 8.15e-03 | **3.35e-03** |

### 5.4 结构对比

比较 ③ 对齐后 SDAA 与 ② CUDA 的 CA 原子坐标（以 ② 为参考）：

- **叠合 RMSD**：两个结构先做最优叠合（Kabsch）再比 —— 只看折叠形状是否一致；
- **未叠合 RMSD**：不做叠合、直接比原始坐标系 —— 形状、全局位置、朝向一起比，更严格。

| 测例 | nCA | 叠合 RMSD | 未叠合 RMSD |
| ---- | --- | --------- | ----------- |
| 1pgb | 56  | 0.004 Å   | 0.264 Å     |
| 1csp | 67  | 0.004 Å   | 0.034 Å     |
| 1ubq | 76  | 0.010 Å   | 0.105 Å     |
| 2ci2 | 83  | 0.029 Å   | 0.049 Å     |
| 1hhp | 99  | 0.008 Å   | 0.041 Å     |
| 1hel | 129 | 0.003 Å   | 0.054 Å     |
| 2lzm | 164 | 0.003 Å   | 0.021 Å     |
| 1ake | 214 | 0.010 Å   | 0.047 Å     |
| 1gfl | 238 | 0.008 Å   | 0.072 Å     |
| 1tim | 247 | 0.004 Å   | 0.020 Å     |
| 1ald | 363 | 0.004 Å   | 0.028 Å     |

11 例的叠合 RMSD 都 ≤ 0.029 Å、未叠合都 ≤ 0.264 Å，**两平台产出的是同一个结构**。

参照：同样这 11 例让 CUDA 自己换一个 seed（42 → 7）重跑，叠合 RMSD 0.102 ~ 2.761 Å、  
未叠合 8.746 ~ 32.322 Å —— 对齐后的 SDAA 比 CUDA 自己的两次运行还接近。**未叠合 RMSD 是关键**：  
换 seed 时坐标系是自由量（8 ~ 32 Å），而 ③ 只有 0.020 ~ 0.264 Å，说明两平台连全局坐标系都几乎重合，  
而不只是「折叠形状一样」。

## 6. 性能与峰值显存

11 个测试蛋白（56 ~ 363 aa）在两侧各跑一遍，**均单卡单例**（`--accelerator gpu|sdaa --devices 1`，  
SDAA 侧每例独占 1 张卡），共用 `--sampling_steps 50 --recycling_steps 3 --seed 42 --max_msa_seqs 2048`。  
耗时取「预测阶段」（`Trainer.predict` 墙钟，不含 python 导入、输入预处理与权重加载）。

| 测例     | 长度   | MSA 条数 | CUDA 耗时 (s) | SDAA 耗时 (s) | 倍数      |
| -------- | ------ | -------- | ------------- | ------------- | --------- |
| 1pgb     | 56 aa  | 144      | 4.93          | 19.34         | **3.9×**  |
| 1csp     | 67 aa  | 9518     | 5.43          | 26.50         | **4.9×**  |
| 1ubq     | 76 aa  | 9655     | 5.14          | 30.76         | **6.0×**  |
| 2ci2     | 83 aa  | 2190     | 5.45          | 33.54         | **6.2×**  |
| 1hhp     | 99 aa  | 4953     | 5.40          | 45.45         | **8.4×**  |
| 1hel     | 129 aa | 3947     | 5.45          | 71.71         | **13.2×** |
| 2lzm     | 164 aa | 1180     | 5.74          | 103.94        | **18.1×** |
| 1ake     | 214 aa | 7627     | 6.95          | 166.45        | **23.9×** |
| 1gfl     | 238 aa | 352      | 7.09          | 192.05        | **27.1×** |
| 1tim     | 247 aa | 7470     | 8.07          | 204.68        | **25.4×** |
| 1ald     | 363 aa | 4083     | 12.50         | 434.49        | **34.8×** |
| **合计** |        |          | **72.2**      | **1328.9**    | **18.4×** |

倍数随规模增长：56 aa 的 1pgb 为 3.9×，363 aa 的 1ald 为 34.8×。

峰值显存（torch 侧 `max_memory_allocated`）：SDAA 侧 2.0 ~ 7.4 GB（最小 1pgb、最大 1ald），  
约为 CUDA 侧的 1.0 ~ 1.6 倍。**SDAA 上 autocast 被禁用、实际以 fp32 运行，CUDA 侧为 bf16 AMP**（见第 7 节），  
这是两侧显存差异的主因。

## 7. 已知限制

- **设备 RNG 对齐是随机数算法问题**：SDAA 出厂的设备 generator（TecoRAND）与 CUDA（Philox）用的是  
  不同的随机数算法，同 seed 抽出的不是同一条序列，所以需要第 3 节的  
  `TORCH_SDAA_ALIGN_NV_DEVICE=a100` 把两侧拉到同一条序列上，与模型本身无关。  
  当前只在 `boltz predict` **单卡**（`--devices 1`）路径验证过；  
  **多卡（DDP，`start_method="fork"`）与训练侧未验证**。
- **`nn.Embedding` rank≥3 依赖本地规避**：2 节中的 `embedding_lookup` 是对 Torch-SDAA 3.2.0  
  算子缺陷的规避，不是模型问题。最小复现见 `sdaa/repro_embedding_sdaa.py`，上报材料见  
  `sdaa/ISSUE_tecodnn_embedding_rank3.md`；算子修复后应移除该规避。
- **autocast 被禁用，实际 fp32 运行**：日志先打印 "Using bfloat16 AMP"，但 boltz 走 CUDA 路径的 `torch.autocast("cuda", bfloat16)` 在 SDAA 上因 `torch.cuda.is_available()==False` 被自动 Disable（伴随 UserWarning），实测为 no-op（autocast 包裹下的 sdaa 张量 matmul 输出 fp32）。`torch.autocast("sdaa", bfloat16)` 本身可用；如需对齐 bf16 性能需将 autocast device_type 按设备分发并重新验证精度，暂未启用。对结果数值无影响（fp32 精度更高），但速度/显存与 CUDA 基线对比时需注明此差异。
- 训练/微调：官方尚未开放 Boltz2 训练代码，未适配。
- MSA server 自动生成（`--use_msa_server`）依赖外部网络服务，未验证；本地 a3m / `msa: empty` 方式不受影响。官方不带 msa 字段的测例（prot.yaml、cyclic_prot.yaml 等）离线环境无法直接跑。
- 长序列 / 大复合物推理耗时随 token 数增长，本次 11 个测例里最长的 363 aa 单例在 SDAA 上耗时 7.2 min。
