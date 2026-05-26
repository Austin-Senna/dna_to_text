"""Translated amino-acid k-mer composition baseline.

Translates the CDS to protein (standard table, stop-truncated) and counts
amino-acid k-mers over the 20-letter alphabet: k=1 -> 20-dim, k=2 -> 400-dim,
k=3 -> 8000-dim. L1-normalised. k-mers containing the unknown placeholder
('X', from non-ACGT codons) are skipped. This control asks whether the
encoded protein's residue composition alone recovers family labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from protein import AMINO_ACIDS, translate_cds
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_DIR = REPO_ROOT / "data" / "sequences"

# Map each amino acid to 0..19; everything else (e.g. 'X') to 255 (invalid).
_AA_LOOKUP = np.full(256, 255, dtype=np.uint8)
for _i, _aa in enumerate(AMINO_ACIDS):
    _AA_LOOKUP[ord(_aa)] = _i


def aa_kmer_dim(k: int) -> int:
    return 20 ** k


def featurize_aa_kmer(sequence: str, k: int = 2) -> np.ndarray:
    """Return a (20**k,) float32 L1-normalised AA k-mer frequency vector."""
    if k < 1:
        raise ValueError("k must be >= 1")
    dim = 20 ** k
    protein = translate_cds(sequence, to_stop=True)
    if len(protein) < k:
        return np.zeros(dim, dtype=np.float32)

    bytes_ = np.frombuffer(protein.encode("ascii"), dtype=np.uint8)
    idx = _AA_LOOKUP[bytes_]                 # (L,) each in {0..19, 255}
    valid = idx < 20

    n_windows = len(idx) - k + 1
    if n_windows <= 0:
        return np.zeros(dim, dtype=np.float32)
    kmer_idx = np.zeros(n_windows, dtype=np.int64)
    valid_window = np.ones(n_windows, dtype=bool)
    for j in range(k):
        sl = idx[j:j + n_windows]
        kmer_idx = kmer_idx * 20 + sl.astype(np.int64)
        valid_window &= valid[j:j + n_windows]

    kmer_idx = kmer_idx[valid_window]
    counts = np.bincount(kmer_idx, minlength=dim).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def load_aa_kmer_features(
    name: Literal["train", "val", "test"],
    k: int = 2,
    sequences_dir: Path = SEQUENCES_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_aa, Y, meta) for the named split. X shape: (n, 20**k)."""
    _, Y, meta = load_split(name)
    X = np.zeros((len(meta), 20 ** k), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        X[i] = featurize_aa_kmer(seq, k=k)
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s) in split {name!r}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return X, Y, meta
