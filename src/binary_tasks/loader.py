"""Load X, y_binary, meta for a frozen binary subset built by make_subsets.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from binary_tasks.make_subsets import BinaryTask, BINARY_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

SplitName = Literal["train", "val", "test"]
META_COLUMNS = ["ensembl_id", "symbol", "family", "summary"]


def _subset_path(task: BinaryTask, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"binary_{task}.json"


def load_binary_split(
    task: BinaryTask,
    name: SplitName,
    dataset_path: str | Path,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y_binary, meta) for one split of a frozen binary subset.

    `dataset_path` selects which encoder's X is used (e.g. dataset.parquet for
    DNABERT-2, dataset_nt_v2.parquet for NT-v2). y_binary is int8: 1 if the
    gene is in `positive_ensembl_ids`, 0 otherwise.
    """
    if name not in ("train", "val", "test"):
        raise ValueError(f"unknown split name: {name!r}")
    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")

    payload = json.loads(_subset_path(task, data_dir).read_text())
    ids = payload["split"][name]
    pos_ids = set(payload["positive_ensembl_ids"])

    df = pd.read_parquet(dataset_path)
    df = df.set_index("ensembl_id").loc[ids].reset_index()

    X = np.stack(df["x"].to_numpy()).astype(np.float32)
    y = np.array([1 if eid in pos_ids else 0 for eid in df["ensembl_id"]], dtype=np.int8)
    meta = df[META_COLUMNS].reset_index(drop=True)
    return X, y, meta
