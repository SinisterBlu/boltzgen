# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning DDP strategy for multiple Intel Gaudi HPU devices.

Uses HPUParallelStrategy from lightning_habana (the Habana-maintained
Lightning integration) which wraps PyTorch DDP over the HCCL collective
communication backend — the Habana equivalent of NCCL.

Usage in predict.py:
    strategy = HPUDDPStrategy(precision_plugin=precision_plugin)
    trainer = Trainer(accelerator=HPUAccelerator(), strategy=strategy,
                      devices=N, ...)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from lightning_habana import HPUParallelStrategy as _HPUParallelStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .hpu_precision import HPUMixedPrecision

if TYPE_CHECKING:
    from lightning_fabric.plugins import CheckpointIO
    from lightning_fabric.plugins.precision import Precision


class HPUDDPStrategy(_HPUParallelStrategy):
    """DDP strategy for multiple Intel Gaudi HPU devices (HCCL backend).

    Wraps lightning_habana.HPUParallelStrategy with the same guard/interface
    as SingleHPUStrategy so predict.py can switch between them transparently.

    Static shape bucketing is applied in batch_to_device, identical to
    SingleHPUStrategy, so multi-card runs benefit from the same recipe-cache
    optimisation as single-card runs.
    """

    strategy_name = "hpu_ddp"

    def __init__(
        self,
        precision_plugin: Precision | None = None,
        checkpoint_io: CheckpointIO | None = None,
        **kwargs,
    ) -> None:
        if not (hasattr(torch, "hpu") and torch.hpu.is_available()):
            msg = "`HPUDDPStrategy` requires HPU devices to run"
            raise MisconfigurationException(msg)
        super().__init__(
            precision_plugin=precision_plugin,
            checkpoint_io=checkpoint_io,
            process_group_backend="hccl",
            **kwargs,
        )

    def batch_to_device(
        self,
        batch: Any,
        device: Optional[torch.device] = None,
        dataloader_idx: int = 0,
    ) -> Any:
        """Move batch to HPU then snap tensor shapes to static buckets.

        Mirrors the identical override in SingleHPUStrategy so that multi-card
        DDP runs get the same static-shape bucketing benefit as single-card runs.
        Each DDP rank pads its own shard of the batch independently before
        forwarding — this is correct because all ranks always receive shards of
        the same bucket size (DataLoader already distributed uniformly).
        """
        batch = super().batch_to_device(batch, device, dataloader_idx)

        use_static = os.environ.get("BOLTZGEN_HPU_STATIC_SHAPES", "1") == "1"
        if use_static:
            from boltzgen.utils.hpu.static_shapes import get_atom_buckets, get_token_buckets, pad_batch_to_buckets

            if not getattr(self, "_buckets_logged", False):
                self._buckets_logged = True
                # Only log on rank 0 to avoid 8× duplicate lines
                if self.global_rank == 0:
                    print(
                        f"[STATIC_SHAPES] token_buckets={get_token_buckets()} "
                        f"atom_buckets={get_atom_buckets()} (DDP rank 0)",
                        flush=True,
                    )
            batch = pad_batch_to_buckets(batch)

        return batch
