"""4-mer frequency featuriser for CDS sequences.

Sliding window of length 4, stride 1. Alphabet = ACGT (256 possible 4-mers,
lex-ordered via base-4 index). Windows containing any non-ACGT base are
skipped. Output is L1-normalised so genes of different CDS lengths are
comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_DIR = REPO_ROOT / "data" / "sequences"

_LOOKUP = np.full(256, 255, dtype=np.uint8)
_LOOKUP[ord("A")] = 0
_LOOKUP[ord("C")] = 1
_LOOKUP[ord("G")] = 2
_LOOKUP[ord("T")] = 3

KMER_DIM = 256  # 4^4, the original CDS 4-mer baseline


def featurize_kmer(sequence: str, k: int = 4) -> np.ndarray:
    """Return a (4**k,) float32 L1-normalised k-mer frequency vector.

    Sliding window of length ``k``, stride 1, over the ACGT alphabet (base-4
    lex index). Windows containing any non-ACGT base are skipped. Generalises
    the original hardcoded 4-mer featuriser so higher-order baselines (e.g.
    the 6-mer composition control) reuse the same code path.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    dim = 4 ** k
    if len(sequence) < k:
        return np.zeros(dim, dtype=np.float32)

    bytes_ = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    idx = _LOOKUP[bytes_]                # (L,) each in {0,1,2,3,255}
    valid = idx < 4

    # Build the base-4 k-mer index and the per-window validity mask by summing
    # shifted views, mirroring the original unrolled 4-mer computation.
    n_windows = len(idx) - k + 1
    if n_windows <= 0:
        return np.zeros(dim, dtype=np.float32)
    kmer_idx = np.zeros(n_windows, dtype=np.int64)
    valid_window = np.ones(n_windows, dtype=bool)
    for j in range(k):
        sl = idx[j:j + n_windows]
        kmer_idx = kmer_idx * 4 + sl.astype(np.int64)
        valid_window &= valid[j:j + n_windows]

    kmer_idx = kmer_idx[valid_window]
    counts = np.bincount(kmer_idx, minlength=dim).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def featurize_sequence(sequence: str) -> np.ndarray:
    """Return a (256,) float32 L1-normalised 4-mer frequency vector."""
    return featurize_kmer(sequence, k=4)


def featurize_cds(sequence: str) -> np.ndarray:
    """Backward-compatible alias for CDS 4-mer features."""
    return featurize_sequence(sequence)


def load_kmer_features(
    name: Literal["train", "val", "test"],
    sequences_dir: Path = SEQUENCES_DIR,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_kmer, Y, meta) for the named split. X_kmer shape: (n, 4**k)."""
    _, Y, meta = load_split(name)
    X = np.zeros((len(meta), 4 ** k), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        X[i] = featurize_kmer(seq, k=k)
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s) in split {name!r}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return X, Y, meta
