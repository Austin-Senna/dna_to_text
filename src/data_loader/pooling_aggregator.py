"""Across-chunk pooling reductions for the encoder pooling sweep.

Each function takes the per-chunk arrays produced by multi_pool and returns
one fixed-length vector per gene. The probe-stage results are summarized in
docs/stage3-train-probes.md.

Variants:
    meanmean  : mean across chunks of (mean tokens per chunk).        d
    centermean: the center chunk's (mean tokens) vector.              d
    specialmean: mean across chunks of (mean all tokens per chunk).   d
    maxmean   : mean across chunks of (max  tokens per chunk).        d
    clsmean   : mean across chunks of (CLS  per chunk).               d
    meanD     : concat[first, last, mean] of (mean tokens).           3d
    meanG     : concat[first, last, mean, max] of (mean tokens).      4d

`centermean` takes the chunk at index ``n_chunks // 2``, a count-based midpoint
and NOT the TSS. Chunks overlap by ``stride`` and the last chunk is aligned to
the sequence end, so this chunk sits systematically 3' of the window's bp
midpoint: median offset +1.2 kb (DNABERT-2), +1.4 kb (GENA-LM), +5.6 kb (NT-v2,
HyenaDNA). It contains the TSS base in ~50% of genes for the 510-token encoders
but only 0.03% for the wide-window ones, so it is a TSS-*downstream* chunk and
is not a same-locus comparison across encoders. Measured in
``analysis/tss_overlap/center_chunk_finding.md`` (``check_tss_center_chunk.py``).
For a genuine TSS-anchored feature (``argmin_k |chunk_bp_center - TSS|``) use
``scripts/build_tss_anchored_datasets.py``.

The "max" inside meanG is per-dim max ACROSS chunks of the mean-tokens-per-
chunk vectors — distinct from maxmean (which is mean ACROSS chunks of the
max-tokens-within-chunk vectors).
"""
from __future__ import annotations

import numpy as np

POOLING_VARIANTS = ("meanmean", "centermean", "specialmean", "maxmean", "clsmean", "meanD", "meanG")


def available_variants(per_chunk: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Return pooling variants supported by the available per-chunk reductions."""
    variants = ["meanmean", "centermean"]
    if "special_mean" in per_chunk:
        variants.append("specialmean")
    if "max" in per_chunk:
        variants.append("maxmean")
    if "cls" in per_chunk:
        variants.append("clsmean")
    variants.extend(["meanD", "meanG"])
    return tuple(variants)


def aggregate(per_chunk: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """Reduce per-chunk arrays to a single fixed-length vector per gene."""
    mean = per_chunk["mean"]   # (n_chunks, d)
    if variant == "meanmean":
        return mean.mean(axis=0).astype(np.float32)
    if variant == "centermean":
        return mean[mean.shape[0] // 2].astype(np.float32)
    if variant == "specialmean":
        if "special_mean" not in per_chunk:
            raise ValueError("specialmean requested but per-chunk reductions do not include 'special_mean'")
        return per_chunk["special_mean"].mean(axis=0).astype(np.float32)
    if variant == "maxmean":
        return per_chunk["max"].mean(axis=0).astype(np.float32)
    if variant == "clsmean":
        if "cls" not in per_chunk:
            raise ValueError("clsmean requested but per-chunk reductions do not include 'cls'")
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
    if variant in ("meanmean", "centermean", "specialmean", "maxmean", "clsmean"):
        return per_chunk_d
    if variant == "meanD":
        return 3 * per_chunk_d
    if variant == "meanG":
        return 4 * per_chunk_d
    raise ValueError(f"unknown pooling variant: {variant!r}")
