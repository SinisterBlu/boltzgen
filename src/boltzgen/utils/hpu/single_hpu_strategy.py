"""Lightning strategy for single Intel Gaudi HPU device."""

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.strategies import SingleDeviceStrategy, StrategyRegistry
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training on a single Intel Gaudi HPU device."""

    strategy_name = "hpu_single"

    def __init__(
        self,
        device: _DEVICE = "hpu:0",
        accelerator: pl.accelerators.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
    ) -> None:
        if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
            msg = "`SingleHPUStrategy` requires HPU devices to run"
            raise MisconfigurationException(msg)

        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )


StrategyRegistry.register(
    SingleHPUStrategy.strategy_name,
    SingleHPUStrategy,
    description="Strategy that enables training on a single Intel Gaudi HPU",
)
