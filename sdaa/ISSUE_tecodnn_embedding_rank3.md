# [BUG] `aten::embedding` / `F.embedding` 在索引 rank≥3 时输出未初始化内存

**组件**：TecoDNN 3.2.0（`aten::embedding`）/ Torch-SDAA 3.2.0
**平台**：LoongArch node499（`10.71.15.115`），`/dev/tcaicard0`，PyTorch 2.7.1
**严重度**：高 —— 静默算错，不报错、不崩，只让结果偏
**复现**：`repro_embedding_sdaa.py`，纯 torch，无 checkpoint，约 2 秒

## 现象

`F.embedding(input, weight)`（`nn.Embedding.forward` 的唯一实现）在
**`input.dim() >= 3`** 时：只正确写入了输出张量的前 `prod(input.shape[:-1])` 行，
其余行保留**未初始化设备内存**。

`input.dim() <= 2` 时一切正常。

## 最小复现

参考基准取 CPU 上的精确索引 `W[idx]`（CPU 侧对所有形状都逐位正确）：

```
idx.shape        device   wrong positions   first wrong flat idx
(4, 4)           sdaa     0/16              -              <- rank 2 正常
(1, 4, 4)        sdaa     9/16              4              <- = N
(2, 4, 4)        sdaa     22/32             9              <- ≈ B*N = 8
(1, 64, 64)      sdaa     4032/4096         64             <- = N
(1, 117, 117)    sdaa     13572/13689       117            <- = N
(1, 1, 4, 4)     sdaa     14/16             1              <- rank 4 同样坏
```

规律上，**rank-3 的 `(B, N, N)` 被当成 `(B*N, N)` 处理，只写了前 `B*N` 行**；
rank-4 更糟 —— 实测 `(1,1,4,4)` 从第 1 行起就错。

> **注（复现计数会漂移）**：上表计数在多次运行间有小幅波动 —— 未初始化内存里偶有值
> 碰巧等于正确值。2026-09-21 重测 `(1,117,117)` 为 `13568/13689`（首发记录是 `13572`），
> `(2,4,4)` 首个错位是 9 而非理论上的 8。这与「未初始化内存」的判定自洽，
> 也意味着**单元测试不能依赖精确计数**，只判「有没有错位」。

错的那部分值是非零垃圾、含 NaN；同一进程内两次调用逐位相同，**跨进程不同** ——
未初始化内存的典型特征（也意味着单元测试若只跑一遍很容易漏掉）。

## 正确性对照（同一份权重、同一份索引，SDAA 上 9 种实现）

| 实现 | rank=2 | rank≥3 |
|---|---|---|
| `F.embedding` / `nn.Embedding` | ✅ | ❌ |
| `torch.index_select` | ✅ | ✅ |
| `weight[idx]` / `weight[idx.reshape(-1)]` | ✅ | ✅ |
| `one_hot @ weight` | ✅ | ✅ |
| `gather` | ✅ | ✅ |

## 期望行为

`F.embedding` 对任意 rank 的 `input` 都应等价于
`torch.index_select(weight, 0, input.reshape(-1)).reshape(*input.shape, weight.shape[-1])`
（`padding_idx` / `max_norm` 语义另计）。

## 影响面

**任何把带 batch/序列维的整型张量喂给 `nn.Embedding` 的模型都会静默算错。**
本次在一个蛋白结构预测模型（Boltz2）上定位到 5 处命中，其中 1 处在主干：

```
boltz2.py:431         idx(1,117,117) w(7,128)   -> z_init 被污染，主干 s/z 偏 14.5% / 36.9%
confidencev2.py:171   idx(1,117,117) w(7,128)   -> confidence 头 z
confidencev2.py:203   idx(1,117,117) w(64,128)  -> confidence 头（距离直方图 embedding）
confidence.py:295     idx(1,117,117) w(64,128)  -> confidence v1 头
affinity.py:109       idx(1,117,117) w(64,128)  -> affinity 头
```

把这一处换成 `index_select` 后，主干全链从 `1e-1 ~ 3.7e-1` 的偏差塌回
`1e-6 ~ 1e-7`（fp32 舍入噪声级别），说明这就是唯一根因。

同类模型里已静态定位到命中的还有 ESMFold / ESM-MSA 的 3 处
（`residue_index` 两两差值、`recycle_bins`、MSA 的 `(B,R,C)` token 张量）。

## 建议

1. 修 `aten::embedding` 的 rank≥3 分支：rank>2 时按 `(prod(shape[:-1]), shape[-1])`
   二维化索引、写出全部行，或直接转发到 `index_select`。
2. 补一条 rank≥3 的单元测试用例（`(1,4,4)` / `(2,4,4)` / `(1,1,4,4)`），
   并对同一输入重复调用多次，避免未初始化内存「碰巧为 0」导致假通过。
3. 若短期无法修，建议在 Torch-SDAA 的 `F.embedding` Python 层加 rank≥3 的
   `index_select` 转发，避免所有下游模型各自 workaround。

## 客户侧规避

**已在 Boltz2 源码内落地**（见 `sdaa/SDAA_ADAPTATION.md` 5.4），共替换 5 处调用点：

```python
# src/boltz/model/modules/utils.py   [sdaa-adapt]
def embedding_lookup(embedding, indices):
    if (indices.device.type == "sdaa" and indices.dim() >= 3
            and embedding.padding_idx is None and embedding.max_norm is None):
        weight = embedding.weight
        out = torch.index_select(weight, 0, indices.reshape(-1))
        return out.reshape(*indices.shape, weight.shape[-1])
    return embedding(indices)
```

rank≤2 与非 sdaa 设备仍走原始调用。实测：源码规避与运行期 patch 的 confidence
输出 9 项指标逐位相同；主干偏差 `1e-1~3.7e-1` → `1e-6~1e-7`。

其它模型可零改动挂载 `sdaa_embedding_guard.py`（import 即生效）：

```python
import sdaa_embedding_guard   # 把 rank>=3 的 aten::embedding 改写成 index_select
```
