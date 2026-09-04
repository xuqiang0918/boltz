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
