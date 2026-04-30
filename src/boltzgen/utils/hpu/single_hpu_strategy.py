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
import warnings
from typing import Any, Callable, Dict, Union

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
        self._maybe_patch_fusedsdpa()
        self._maybe_wrap_hpu_graphs()

    def _maybe_patch_fusedsdpa(self) -> None:
        """Replace F.scaled_dot_product_attention with Habana FusedSDPA.

        gpu_migration does NOT automatically redirect SDPA → FusedSDPA, so
        every attention call falls back to the CPU reference implementation
        (aten::_scaled_dot_product_attention_math).  FusedSDPA is Gaudi's
        Flash-Attention equivalent: fully fused kernel, lower memory, same
        BF16/FP32 interface.

        The patch is applied globally to torch.nn.functional but only routes
        HPU tensors to FusedSDPA; CPU/CUDA tensors continue to use the
        original implementation unchanged.

        Controlled by BOLTZGEN_HPU_FUSEDSDPA (default: 1 = enabled).
        """
        if os.environ.get("BOLTZGEN_HPU_FUSEDSDPA", "1") != "1":
            return
        try:
            from habana_frameworks.torch.hpex.kernels import FusedSDPA
            import torch.nn.functional as F

            _orig_sdpa = F.scaled_dot_product_attention

            def _hpu_sdpa(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale=None,
                **kwargs,
            ):
                if query.device.type == "hpu":
                    return FusedSDPA.apply(
                        query, key, value, attn_mask, dropout_p, is_causal, scale
                    )
                return _orig_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )

            F.scaled_dot_product_attention = _hpu_sdpa
            print(
                "[SingleHPUStrategy] Patched F.scaled_dot_product_attention → FusedSDPA",
                flush=True,
            )
        except ImportError:
            warnings.warn(
                "FusedSDPA not available (habana_frameworks not found). "
                "Attention will use CPU math fallback.",
                stacklevel=2,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"FusedSDPA patch failed: {exc}. "
                "Attention will use CPU math fallback.",
                stacklevel=2,
            )

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
        """Run validation step then flush lazy graph."""
        import habana_frameworks.torch.core as htcore
        result = super().validation_step(*args, **kwargs)
        htcore.mark_step()
        return result

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Run test step then flush lazy graph."""
        import habana_frameworks.torch.core as htcore
        result = super().test_step(*args, **kwargs)
        htcore.mark_step()
        return result

    # Class-level counter so the profiler fires only on the first N steps.
    _profile_steps_done: int = 0

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """Run predict step then flush lazy graph once per batch.

        Calling mark_step() AFTER super() ensures the full forward-pass graph
        is accumulated before it is dispatched to the HPU.  Calling it before
        (as was done previously) caused every step to flush an incomplete graph,
        triggering a fresh compilation for each dynamic shape encountered.

        Optional profiling: set ``BOLTZGEN_PROFILE_STEPS=N`` (env var) to wrap
        the first N predict_steps with ``torch.profiler``.  Outputs:
          • /app/benchmarks/profiles/design_trace_<N>.json  (Chrome/Perfetto)
          • printed per-op CPU-time table (top 40 ops)
        Open the JSON at https://ui.perfetto.dev for a flame graph.
        """
        import habana_frameworks.torch.core as htcore

        profile_n = int(os.environ.get("BOLTZGEN_PROFILE_STEPS", "0"))
        if profile_n > 0 and SingleHPUStrategy._profile_steps_done < profile_n:
            SingleHPUStrategy._profile_steps_done += 1
            step_idx = SingleHPUStrategy._profile_steps_done
            trace_path = f"/app/benchmarks/profiles/design_trace_{step_idx}.json"
            os.makedirs("/app/benchmarks/profiles", exist_ok=True)
            print(f"[PROFILER] Capturing predict_step {step_idx} → {trace_path}", flush=True)

            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            with prof:
                result = super().predict_step(*args, **kwargs)
            htcore.mark_step()

            prof.export_chrome_trace(trace_path)
            table = prof.key_averages(group_by_stack_n=5).table(
                sort_by="cpu_time_total", row_limit=40
            )
            print(f"\n[PROFILER] Top ops by CPU time (step {step_idx}):\n{table}", flush=True)
            print(f"[PROFILER] Chrome trace → {trace_path}", flush=True)
            print(f"[PROFILER] Open at https://ui.perfetto.dev", flush=True)
            return result

        result = super().predict_step(*args, **kwargs)
        htcore.mark_step()
        return result

    def transfer_batch_to_device(
        self,
        batch: Dict,
        device: torch.device,
        dataloader_idx: int,
    ) -> Dict:
        """Move batch to HPU then snap tensor shapes to static buckets.

        Static-shape bucketing eliminates per-design HPU graph recompilation:
        instead of one compiled recipe per unique (n_tokens, n_atoms) pair,
        we compile exactly O(n_token_buckets) recipe sets — total.  After the
        first warm-up run for a given bucket size, all subsequent designs of
        that length are instant cache hits.

        Controlled by ``BOLTZGEN_HPU_STATIC_SHAPES`` (default: 1 when HPU
        is active).  Bucket sizes: ``BOLTZGEN_HPU_TOKEN_BUCKETS`` and
        ``BOLTZGEN_HPU_ATOM_BUCKETS`` env vars (comma-separated ints).
        """
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        use_static = os.environ.get("BOLTZGEN_HPU_STATIC_SHAPES", "1") == "1"
        if use_static:
            from boltzgen.utils.hpu.static_shapes import pad_batch_to_buckets
            batch = pad_batch_to_buckets(batch)

        return batch


StrategyRegistry.register(
    SingleHPUStrategy.strategy_name,
    SingleHPUStrategy,
    description="Strategy that enables training on a single Intel Gaudi HPU",
)
