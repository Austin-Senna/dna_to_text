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

KMER_DIM = 256


def featurize_cds(sequence: str) -> np.ndarray:
    """Return a (256,) float32 L1-normalised 4-mer frequency vector."""
    if len(sequence) < 4:
        return np.zeros(KMER_DIM, dtype=np.float32)

    bytes_ = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    idx = _LOOKUP[bytes_]                # (L,) each in {0,1,2,3,255}
    valid = idx < 4
    b0, b1, b2, b3 = idx[:-3], idx[1:-2], idx[2:-1], idx[3:]
    valid_window = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]

    kmer_idx = b0 * 64 + b1 * 16 + b2 * 4 + b3
    kmer_idx = kmer_idx[valid_window].astype(np.int64)
    counts = np.bincount(kmer_idx, minlength=KMER_DIM).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def load_kmer_features(
    name: Literal["train", "val", "test"],
    sequences_dir: Path = SEQUENCES_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_kmer, Y, meta) for the named split. X_kmer shape: (n, 256)."""
    _, Y, meta = load_split(name)
    X = np.zeros((len(meta), KMER_DIM), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        X[i] = featurize_cds(seq)
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s) in split {name!r}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return X, Y, meta
