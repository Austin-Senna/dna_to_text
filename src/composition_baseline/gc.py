"""Length + GC + coding-length baseline (3-dim).

A deliberately minimal scalar-composition control: GC fraction, log CDS
length, and log codon count (coding length). Catches the "encoder is just a
length/GC proxy" failure mode and is the weakest baseline by construction.
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

GC_DIM = 3


def featurize_gc(sequence: str) -> np.ndarray:
    """Return (3,) float32: [gc_fraction, log1p(len), log1p(len/3)]."""
    s = sequence.upper()
    n = len(s)
    if n == 0:
        return np.zeros(GC_DIM, dtype=np.float32)
    bytes_ = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    g = int((bytes_ == ord("G")).sum())
    c = int((bytes_ == ord("C")).sum())
    a = int((bytes_ == ord("A")).sum())
    t = int((bytes_ == ord("T")).sum())
    acgt = a + c + g + t
    gc_frac = (g + c) / acgt if acgt > 0 else 0.0
    return np.array(
        [gc_frac, np.log1p(n), np.log1p(n / 3.0)], dtype=np.float32
    )


def load_gc_features(
    name: Literal["train", "val", "test"],
    sequences_dir: Path = SEQUENCES_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_gc, Y, meta) for the named split. X shape: (n, 3)."""
    _, Y, meta = load_split(name)
    X = np.zeros((len(meta), GC_DIM), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        X[i] = featurize_gc(seq)
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s) in split {name!r}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return X, Y, meta
