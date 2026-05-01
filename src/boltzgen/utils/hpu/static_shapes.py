"""HPU static-shape bucketing for BoltzGen inference batches.

BoltzGen's diffusion model is shape-sensitive: every unique (n_tokens,
n_atoms) combination that reaches the HPU graph compiler produces a new
compiled recipe.  With purely dynamic shapes this means:

  - First design in a session: compile every unique shape across 200
    diffusion timesteps → 20-40 minutes of cold-start
  - Designs with different sequence lengths: recompile from scratch
  - Recipe cache grows unboundedly

Static-shape bucketing fixes this by padding every batch to a small set
of pre-defined sizes BEFORE the HPU sees the tensors.  The model already
carries ``token_pad_mask`` and ``atom_pad_mask`` tensors that gate all
attention and loss operations, so padding with zeros is lossless.

With buckets = [64, 128, 256, 384, 512, 1024]:
  - A 50-residue protein → padded to 64 tokens → ONE compiled recipe set
  - A 150-residue protein → padded to 256 tokens → same recipe on second run
  - Total unique shapes per model: O(n_buckets) instead of O(n_designs)

Usage
-----
The bucketing is applied automatically by SingleHPUStrategy when
``BOLTZGEN_HPU_STATIC_SHAPES=1`` is set (default: 1 when HPU is active).

Bucket sizes can be overridden:
  BOLTZGEN_HPU_TOKEN_BUCKETS=64,128,256,384,512,1024
  BOLTZGEN_HPU_ATOM_BUCKETS=896,1792,3584,7168,14336

Shape Logging
-------------
Set ``BOLTZGEN_SHAPE_LOG=1`` (default: 1) to emit a [SHAPE_LOG] line to
stdout and append a JSON record to ``/tmp/hpu_shape_log.jsonl`` on every
batch.  Each line reports raw dims, assigned buckets, padding overhead,
and whether this bucket combination is a first-time compile trigger.
This lets you identify which designs land in different buckets and tune
the bucket lists to minimise recompilations.

Grep the benchmark log:
  grep SHAPE_LOG /tmp/bench_lazy_*.log
  grep COMPILE_TRIGGER /tmp/bench_lazy_*.log
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Bucket configuration
# ---------------------------------------------------------------------------

# Tuned buckets derived from shape_run_v2 shape logger data (binder_length=50, 1UBQ target).
#
# Observed raw shapes per pipeline step:
#   design:          tokens=126, atoms=1312  (was wasting 27% in 1792 bucket)
#   inverse_folding: tokens=126, atoms=832   (was wasting  7% in 896  bucket — acceptable)
#   design_folding:  tokens=50,  atoms=416   (was wasting 54% in 896  bucket)
#
# Changes vs original defaults:
#   TOKEN: added 160  → covers binder_length=70 (146 tokens, 9% waste vs 24% in 192)
#   ATOM:  added 448  → covers design_folding   (416 atoms,  7% waste vs 54% in 896)
#   ATOM:  added 640  → covers design_folding for binder_length=70 (~560 atoms)
#   ATOM:  added 1344 → covers design step      (1312 atoms, 2% waste vs 27% in 1792)
DEFAULT_TOKEN_BUCKETS: List[int] = [64, 128, 160, 192, 256, 320, 384, 512, 768, 1024]
DEFAULT_ATOM_BUCKETS: List[int] = [448, 640, 896, 1344, 1792, 2688, 3584, 5376, 7168, 10752, 14336]


def _parse_bucket_env(env_var: str, defaults: List[int]) -> List[int]:
    raw = os.environ.get(env_var, "")
    if not raw:
        return defaults
    try:
        return sorted(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError:
        return defaults


def get_token_buckets() -> List[int]:
    return _parse_bucket_env("BOLTZGEN_HPU_TOKEN_BUCKETS", DEFAULT_TOKEN_BUCKETS)


def get_atom_buckets() -> List[int]:
    return _parse_bucket_env("BOLTZGEN_HPU_ATOM_BUCKETS", DEFAULT_ATOM_BUCKETS)


def snap_to_bucket(size: int, buckets: List[int]) -> int:
    """Return the smallest bucket >= size, or the largest bucket if size exceeds all."""
    for b in sorted(buckets):
        if size <= b:
            return b
    return max(buckets)


# ---------------------------------------------------------------------------
# Shape logger
# ---------------------------------------------------------------------------

# Module-level state — persists for the lifetime of the process.
_seen_buckets: Set[Tuple[int, int]] = set()   # (token_bucket, atom_bucket) pairs compiled so far
_call_count: int = 0                           # total transfer_batch_to_device calls
_shape_log_path: str = "/tmp/hpu_shape_log.jsonl"
_t0: float = time.monotonic()                  # process start time for relative timestamps


def _log_shape(
    n_tokens: int,
    token_bucket: int,
    n_atoms: int,
    atom_bucket: int,
) -> None:
    """Log raw dims, assigned buckets, and compile-trigger status.

    Output goes to stdout (grep-friendly) AND to /tmp/hpu_shape_log.jsonl
    for post-run analysis.

    Tokens/atoms overhead = how many padding elements were added (wasted
    compute).  First time a (token_bucket, atom_bucket) pair is seen it
    marks a COMPILE_TRIGGER — expect a long pause after this batch.
    """
    global _call_count
    _call_count += 1

    bucket_key = (token_bucket, atom_bucket)
    is_new = bucket_key not in _seen_buckets
    if is_new:
        _seen_buckets.add(bucket_key)

    tok_pad = token_bucket - n_tokens
    atom_pad = atom_bucket - n_atoms
    tok_pct = 100 * tok_pad / token_bucket if token_bucket else 0
    atom_pct = 100 * atom_pad / atom_bucket if atom_bucket else 0
    tag = "COMPILE_TRIGGER" if is_new else "BUCKET_HIT"
    elapsed = time.monotonic() - _t0

    # Stdout line — survives tqdm CR-overwrite via \n
    print(
        f"[SHAPE_LOG] call={_call_count:04d} t={elapsed:8.1f}s "
        f"tokens={n_tokens}->{token_bucket}(+{tok_pad},{tok_pct:.0f}%) "
        f"atoms={n_atoms}->{atom_bucket}(+{atom_pad},{atom_pct:.0f}%) "
        f"seen={len(_seen_buckets)} {tag}",
        flush=True,
    )

    # JSONL file — append one record per call
    try:
        record = {
            "call": _call_count,
            "elapsed_s": round(elapsed, 2),
            "n_tokens": n_tokens,
            "token_bucket": token_bucket,
            "token_pad": tok_pad,
            "n_atoms": n_atoms,
            "atom_bucket": atom_bucket,
            "atom_pad": atom_pad,
            "is_compile_trigger": is_new,
            "n_unique_buckets": len(_seen_buckets),
        }
        with open(_shape_log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # don't crash inference if log file is unwriteable


# ---------------------------------------------------------------------------
# Batch padding
# ---------------------------------------------------------------------------

# Keys whose values are NOT tensors (lists, strings, dicts) — skip these.
_NON_TENSOR_KEYS = frozenset({
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "activity_name",
    "activity_qualifier",
    "sid",
    "cid",
    "aid",
    "normalized_protein_accession",
    "pair_id",
    "record",
    "id",
    "structure_bonds",
    "extra_mols",
    "structure",
    "tokenized",
    "data_sample_idx",
})


def _pad_tensor(
    t: Tensor,
    n_tokens: int,
    token_bucket: int,
    n_atoms: int,
    atom_bucket: int,
) -> Tensor:
    """Pad a tensor so all token-sized and atom-sized dims reach bucket size."""
    shape = list(t.shape)
    pad_spec: List[int] = []  # torch.nn.functional.pad uses reverse dim order

    needs_pad = False
    for dim_size in reversed(shape):
        if dim_size == n_tokens and token_bucket > n_tokens:
            pad_spec += [0, token_bucket - n_tokens]
            needs_pad = True
        elif dim_size == n_atoms and atom_bucket > n_atoms:
            pad_spec += [0, atom_bucket - n_atoms]
            needs_pad = True
        else:
            pad_spec += [0, 0]

    if not needs_pad:
        return t

    return torch.nn.functional.pad(t, pad_spec, mode="constant", value=0)


def pad_batch_to_buckets(
    batch: Dict,
    token_buckets: Optional[List[int]] = None,
    atom_buckets: Optional[List[int]] = None,
) -> Dict:
    """Pad all tensors in a BoltzGen batch to static bucket sizes.

    Reads ``token_pad_mask`` and ``atom_pad_mask`` from the batch to
    determine the true sequence/atom lengths, snaps each to the smallest
    bucket that fits, then pads every tensor accordingly.

    Parameters
    ----------
    batch : Dict
        Collated batch dict from a BoltzGen DataModule.
    token_buckets : list of int, optional
        Override token bucket sizes (default: from env / DEFAULT_TOKEN_BUCKETS).
    atom_buckets : list of int, optional
        Override atom bucket sizes (default: from env / DEFAULT_ATOM_BUCKETS).

    Returns
    -------
    Dict
        Batch with all token/atom dimensions padded to bucket sizes.
    """
    if token_buckets is None:
        token_buckets = get_token_buckets()
    if atom_buckets is None:
        atom_buckets = get_atom_buckets()

    # Determine n_tokens and n_atoms from mask tensors.
    # token_pad_mask shape: [batch, n_tokens] — 1 for real tokens, 0 for pad
    # atom_pad_mask  shape: [batch, n_atoms]
    token_mask = batch.get("token_pad_mask")
    atom_mask = batch.get("atom_pad_mask")

    if token_mask is None:
        # Nothing to snap — return unchanged
        return batch

    n_tokens: int = token_mask.shape[-1]
    token_bucket: int = snap_to_bucket(n_tokens, token_buckets)

    if atom_mask is not None:
        n_atoms: int = atom_mask.shape[-1]
        atom_bucket: int = snap_to_bucket(n_atoms, atom_buckets)
    else:
        n_atoms = 0
        atom_bucket = 0

    if token_bucket == n_tokens and atom_bucket == n_atoms:
        # Already bucket-aligned — nothing to do
        return batch

    if os.environ.get("BOLTZGEN_SHAPE_LOG", "1") == "1":
        _log_shape(n_tokens, token_bucket, n_atoms, atom_bucket)

    padded: Dict = {}
    for key, value in batch.items():
        if key in _NON_TENSOR_KEYS or not isinstance(value, Tensor):
            padded[key] = value
            continue
        padded[key] = _pad_tensor(value, n_tokens, token_bucket, n_atoms, atom_bucket)

    return padded
