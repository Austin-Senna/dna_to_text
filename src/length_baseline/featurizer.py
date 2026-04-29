"""CDS-length-only baseline. Single feature: log(len(cds)+1).

Catches the "encoder is just a length proxy" failure mode. We use log-length
so the feature is on the same order of magnitude as a typical embedding
dimension and the logistic regression's regularisation is well-scaled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.sequence_fetcher import fetch_cds

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_DIR = REPO_ROOT / "data" / "sequences"

LENGTH_DIM = 1


def cds_length_features(
    meta: pd.DataFrame,
    sequences_dir: Path = SEQUENCES_DIR,
) -> np.ndarray:
    """Return (n, 1) float32 array of log1p(CDS length) for each row of meta."""
    if "ensembl_id" not in meta.columns:
        raise ValueError("meta must have an 'ensembl_id' column")

    out = np.zeros((len(meta), LENGTH_DIM), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        out[i, 0] = np.log1p(len(seq))
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return out
