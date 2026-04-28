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

HPU Graphs (opt-in via BOLTZGEN_HPU_GRAPHS=1):
  The model forward is wrapped with habana_frameworks wrap_in_hpu_graph so
  that the first call per unique input-shape records the HPU op graph and
  all subsequent same-shape calls replay it in hardware.  This eliminates
  Python dispatch overhead for every op in the forward pass.

  For a BoltzGen run the binder length is fixed per experiment, so after one
  compilation hit the same graph is replayed for every design → large speedup.
  Set max_graphs via BOLTZGEN_HPU_MAX_GRAPHS (default 50) to bound memory.

  Requires PT_HPU_LAZY_MODE=1 to be set in the environment.
"""

import os
from typing import Any, Callable, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning import LightningModule, Trainer
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

    Set ``BOLTZGEN_HPU_GRAPHS=1`` to enable HPU graph capture/replay on the
    model forward pass.  Set ``BOLTZGEN_HPU_MAX_GRAPHS`` to limit the number
    of cached shapes (default 50).
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
    # HPU graph setup
    # ------------------------------------------------------------------

    def setup(self, trainer: Trainer) -> None:
        """Move model to HPU then optionally wrap with HPU graph capture."""
        self.model_to_device()
        super().setup(trainer)
        self._maybe_wrap_hpu_graphs()

    def _maybe_wrap_hpu_graphs(self) -> None:
        """Wrap model forward with HPU graph capture/replay if enabled.

        Controlled by env vars:
          BOLTZGEN_HPU_GRAPHS=1       — enable (default: off)
          BOLTZGEN_HPU_MAX_GRAPHS=N   — max cached shape graphs (default: 50)

        First call per unique input shape records the op graph; all
        subsequent same-shape calls replay it from hardware cache, eliminating
        Python dispatch overhead for every tensor op in the forward pass.

        Falls back gracefully if the Habana graphs API is unavailable.
        """
        use_graphs = os.environ.get("BOLTZGEN_HPU_GRAPHS", "0") == "1"
        if not use_graphs:
            return

        model = getattr(self, "model", None)
        if model is None:
            return

        max_graphs = int(os.environ.get("BOLTZGEN_HPU_MAX_GRAPHS", "50"))
        lazy_mode = os.environ.get("PT_HPU_LAZY_MODE", "0") == "1"
        if not lazy_mode:
            import warnings
            warnings.warn(
                "BOLTZGEN_HPU_GRAPHS=1 but PT_HPU_LAZY_MODE is not 1. "
                "HPU graphs require lazy mode.  Set PT_HPU_LAZY_MODE=1.",
                stacklevel=2,
            )
            return

        try:
            from habana_frameworks.torch.hpu.graphs import wrap_in_hpu_graph
            wrapped = wrap_in_hpu_graph(model, max_graphs=max_graphs)
            # Replace the model on the strategy and on the trainer
            self.model = wrapped
            if hasattr(self, "_lightning_module"):
                # pytorch_lightning 2.x keeps a reference here too
                self._lightning_module = wrapped
            print(
                f"[SingleHPUStrategy] HPU graph capture enabled "
                f"(max_graphs={max_graphs}).  First call per shape will compile; "
                "subsequent same-shape calls will replay from cache."
            )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Failed to wrap model with HPU graphs: {exc}. "
                "Continuing without HPU graph capture.",
                stacklevel=2,
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
