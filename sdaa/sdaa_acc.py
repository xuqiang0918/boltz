import os

# [sdaa-adapt] Align the SDAA device generator with the CUDA (Philox) layout.
#
# The stock TecoRAND device generator is not the same sequence as CUDA's, which
# biases the diffusion sampler and collapses the prediction.  The platform
# variable TORCH_SDAA_ALIGN_NV_DEVICE (parsed by libtorch_sdaa.so, init_nv_option)
# swaps it for a Philox-layout one; supported values are a100 / v100.  It is
# enough to set it before the first device RNG consumption, so importing this
# module is early enough -- no change to main.py, no warm-up draw needed.
#
# BOLTZ_SDAA_ALIGN_NV=0 keeps the stock generator (A/B switch used to reproduce
# the defect).  Never set the platform variable itself to 0/off/none: it is
# parsed as a device name and aborts.  Measured results -- three arms, the
# judgement criterion and the operator coverage -- are in
# sdaa/SDAA_ADAPTATION.md section 5.
if os.environ.get("BOLTZ_SDAA_ALIGN_NV", "1") == "0":
    os.environ.pop("TORCH_SDAA_ALIGN_NV_DEVICE", None)
else:
    os.environ.setdefault("TORCH_SDAA_ALIGN_NV_DEVICE", "a100")

import torch
from typing_extensions import override
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

class SDAAAccelerator(CUDAAccelerator):
    """Accelerator for Teco SDAA devices (登临), masquerades as CUDA for PL internals."""

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "sdaa":
            raise MisconfigurationException(f"Device should be SDAA, got {device} instead.")
        torch.sdaa.set_device(device.index if device.index is not None else 0)

    @override
    def setup(self, trainer) -> None:
        pass

    @override
    def get_device_stats(self, device: _DEVICE) -> dict:
        return {}

    @override
    def teardown(self) -> None:
        pass

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

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register("sdaa", cls, description=cls.__name__)
