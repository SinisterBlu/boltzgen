"""HPU utilities.

By importing this module, the accelerator and strategy are registered in lightning.
Importing also triggers habana_frameworks.torch.core initialisation which is
required before any HPU tensor operation.
"""

import habana_frameworks.torch.core  # noqa: F401  # must be first

from .hpu_accelerator import HPUAccelerator
from .hpu_precision import HPUMixedPrecision
from .single_hpu_strategy import SingleHPUStrategy

__all__ = ["HPUAccelerator", "HPUMixedPrecision", "SingleHPUStrategy"]
