"""HPU Accelerator for Intel Gaudi."""

from typing import Any

import torch
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry


class HPUAccelerator(Accelerator):
    """Support for Intel Gaudi HPU, optimized for large-scale machine learning."""

    @property
    def name(self) -> str:
        """Accelerator name required by pytorch-lightning >= 2.5.6."""
        return "hpu"

    @staticmethod
    def setup_device(device: torch.device) -> None:
        """Sets up the specified HPU device."""
        if device.type != "hpu":
            msg = f"Device should be hpu, got {device} instead"
            raise RuntimeError(msg)
        torch.hpu.set_device(device)

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parses devices for multi-HPU training."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list) -> list[torch.device]:
        """Generates a list of parallel HPU devices."""
        return [torch.device("hpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Returns the number of HPU devices available."""
        return torch.hpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Checks if HPU is available."""
        return hasattr(torch, "hpu") and torch.hpu.is_available()

    @staticmethod
    def get_device_stats(device: str | torch.device) -> dict[str, Any]:
        """Returns HPU device stats."""
        del device  # Unused
        return {}

    def teardown(self) -> None:
        """Teardown the HPU accelerator."""


AcceleratorRegistry.register(
    HPUAccelerator().name,
    HPUAccelerator,
    description="Accelerator supports Intel Gaudi HPU devices",
)
