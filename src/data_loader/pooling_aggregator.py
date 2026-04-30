"""Across-chunk pooling reductions for Phase 4b.

Each function takes the per-chunk arrays produced by multi_pool and returns
one fixed-length vector per gene. See findings.md § "Phase 4 — Classification
reframing" and the spec § "Pooling deferred" for the menu.

Variants:
    meanmean  : mean across chunks of (mean tokens per chunk).        d
    maxmean   : mean across chunks of (max  tokens per chunk).        d
    clsmean   : mean across chunks of (CLS  per chunk).               d
    meanD     : concat[first, last, mean] of (mean tokens).           3d
    meanG     : concat[first, last, mean, max] of (mean tokens).      4d

The "max" inside meanG is per-dim max ACROSS chunks of the mean-tokens-per-
chunk vectors — distinct from maxmean (which is mean ACROSS chunks of the
max-tokens-within-chunk vectors).
"""
from __future__ import annotations

import numpy as np

POOLING_VARIANTS = ("meanmean", "maxmean", "clsmean", "meanD", "meanG")


def aggregate(per_chunk: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """Reduce per-chunk arrays to a single fixed-length vector per gene."""
    mean = per_chunk["mean"]   # (n_chunks, d)
    if variant == "meanmean":
        return mean.mean(axis=0).astype(np.float32)
    if variant == "maxmean":
        return per_chunk["max"].mean(axis=0).astype(np.float32)
    if variant == "clsmean":
        return per_chunk["cls"].mean(axis=0).astype(np.float32)
    if variant == "meanD":
        return np.concatenate(
            [mean[0], mean[-1], mean.mean(axis=0)], axis=0
        ).astype(np.float32)
    if variant == "meanG":
        return np.concatenate(
            [mean[0], mean[-1], mean.mean(axis=0), mean.max(axis=0)], axis=0
        ).astype(np.float32)
    raise ValueError(f"unknown pooling variant: {variant!r}")


def output_dim(variant: str, per_chunk_d: int) -> int:
    if variant in ("meanmean", "maxmean", "clsmean"):
        return per_chunk_d
    if variant == "meanD":
        return 3 * per_chunk_d
    if variant == "meanG":
        return 4 * per_chunk_d
    raise ValueError(f"unknown pooling variant: {variant!r}")
