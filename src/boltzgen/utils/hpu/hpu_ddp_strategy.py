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

from typing import TYPE_CHECKING

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
