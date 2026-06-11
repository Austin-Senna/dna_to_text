"""CDS codon-frequency baseline (64-dim).

Reads the CDS in frame 0, counts each of the 64 standard codons, and
L1-normalises. Codons containing any non-ACGT base are skipped. This is the
compositional control that asks whether codon usage alone — not the DNA
language model — recovers protein-family labels.
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

CODON_DIM = 64

_LOOKUP = np.full(256, 255, dtype=np.uint8)
_LOOKUP[ord("A")] = 0
_LOOKUP[ord("C")] = 1
_LOOKUP[ord("G")] = 2
_LOOKUP[ord("T")] = 3


def featurize_codon(sequence: str) -> np.ndarray:
    """Return a (64,) float32 L1-normalised in-frame codon-frequency vector."""
    s = sequence.upper()
    n_codons = len(s) // 3
    if n_codons == 0:
        return np.zeros(CODON_DIM, dtype=np.float32)

    bytes_ = np.frombuffer(s[: n_codons * 3].encode("ascii"), dtype=np.uint8)
    idx = _LOOKUP[bytes_].reshape(n_codons, 3)          # (n_codons, 3)
    valid = (idx < 4).all(axis=1)                       # drop codons with non-ACGT
    code = idx[:, 0] * 16 + idx[:, 1] * 4 + idx[:, 2]   # base-4 codon index
    code = code[valid].astype(np.int64)
    counts = np.bincount(code, minlength=CODON_DIM).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def load_codon_features(
    name: Literal["train", "val", "test"],
    sequences_dir: Path = SEQUENCES_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_codon, Y, meta) for the named split. X shape: (n, 64)."""
    _, Y, meta = load_split(name)
    X = np.zeros((len(meta), CODON_DIM), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        X[i] = featurize_codon(seq)
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s) in split {name!r}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return X, Y, meta
