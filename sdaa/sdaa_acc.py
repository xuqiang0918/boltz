# [sdaa-adapt] SDAA accelerator for Boltz2 -- see sdaa/SDAA_ADAPTATION.md.
#
# Device-RNG alignment to the CUDA (Philox) layout is a runtime setting, not a
# code change: export TORCH_SDAA_ALIGN_NV_DEVICE=a100 before launching, otherwise
# the stock TecoRAND sequence biases the diffusion sampler (doc section 5).
import torch
from typing_extensions import override
from lightning_fabric.accelerators import _AcceleratorRegistry
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

class SDAAAccelerator(CUDAAccelerator):
    """Accelerator for Teco SDAA devices, masquerades as CUDA for PL internals."""

    # 下面 6 个覆写都必需：父类实现走 torch.cuda / num_cuda_devices()，在 SDAA 上
    # 要么返回 0 张卡、要么直接 raise（实测见 SDAA_ADAPTATION.md 第 2 节）。
    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "sdaa":
            raise MisconfigurationException(f"Device should be SDAA, got {device} instead.")
        torch.sdaa.set_device(device.index if device.index is not None else 0)

    @staticmethod
    @override
    def parse_devices(devices):
        if devices is None or devices == "auto":
            return [0]
        if isinstance(devices, int):
            return list(range(devices)) if devices > 0 else [0]
        if isinstance(devices, (list, tuple)):
            return list(devices)
        return [0]

    @staticmethod
    @override
    def get_parallel_devices(devices):
        parsed = SDAAAccelerator.parse_devices(devices)
        assert parsed is not None
        return [torch.device("sdaa", i) for i in parsed]

    @staticmethod
    @override
    def auto_device_count() -> int:
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        return hasattr(torch, "sdaa") and torch.sdaa.is_available()

    # 必须覆写：父类实现用 cls.name() 注册，而 name() 经注册表反查得到 "cuda"，
    # 会把 CUDA accelerator 顶掉。
    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register("sdaa", cls, description=cls.__name__)
