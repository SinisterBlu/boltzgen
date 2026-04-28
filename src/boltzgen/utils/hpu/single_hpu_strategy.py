"""Lightning strategy for single Intel Gaudi HPU device.

This strategy mirrors the mark_step() placement used by the official
lightning_habana SingleHPUStrategy (v1.6.0).  In PT_HPU_LAZY_MODE=1,
ops accumulate in a lazy graph and are only dispatched to the HPU when
htcore.mark_step() is called.  Without these calls the graph grows
unboundedly, causing OOM or stale outputs.

Key lazy-mode placement rules (from optimum-habana and lightning_habana):
  - After forward pass (validation/test/predict step)
  - After backward pass (on_after_backward)
  - After optimizer step
"""

from typing import Any, Callable, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import SingleDeviceStrategy, StrategyRegistry
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Module
from torch.optim.optimizer import Optimizer


class SingleHPUStrategy(SingleDeviceStrategy):
    """Strategy for training and inference on a single Intel Gaudi HPU device.

    Adds ``htcore.mark_step()`` calls at batch boundaries so that ops
    accumulated in lazy mode are flushed to the HPU at the right points.
    Required when ``PT_HPU_LAZY_MODE=1`` (lazy mode), which is necessary
    for HPU graph capture and BF16 performance.
    """

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

    # ------------------------------------------------------------------
    # Lazy-mode graph-flush hooks
    # ------------------------------------------------------------------

    def on_after_backward(self) -> None:
        """Flush lazy graph after backward pass."""
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any],
        model: Union[LightningModule, Module, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Flush lazy graph after optimizer step."""
        import habana_frameworks.torch.core as htcore
        result = super().optimizer_step(optimizer, closure, model, **kwargs)
        htcore.mark_step()
        return result

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Flush lazy graph after validation step."""
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Flush lazy graph after test step."""
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """Flush lazy graph after predict step (inference path)."""
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
        return super().predict_step(*args, **kwargs)


StrategyRegistry.register(
    SingleHPUStrategy.strategy_name,
    SingleHPUStrategy,
    description="Strategy that enables training on a single Intel Gaudi HPU",
)
