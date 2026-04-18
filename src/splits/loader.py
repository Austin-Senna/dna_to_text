"""Shared split loader. Every downstream script reads arrays through here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATASET_PATH = DATA_DIR / "dataset.parquet"
SPLITS_PATH = DATA_DIR / "splits.json"

SplitName = Literal["train", "val", "test"]
META_COLUMNS = ["ensembl_id", "symbol", "family", "summary"]


def _load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_parquet(dataset_path)


def _load_splits_file(splits_path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(splits_path).read_text())


def load_split(
    name: SplitName,
    dataset_path: Path = DATASET_PATH,
    splits_path: Path = SPLITS_PATH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, Y, meta) for the named split, row-aligned."""
    splits = _load_splits_file(splits_path)
    if name not in ("train", "val", "test"):
        raise ValueError(f"unknown split name: {name!r}")
    ids = splits[name]

    df = _load_dataset(dataset_path)
    df = df.set_index("ensembl_id").loc[ids].reset_index()

    X = np.stack(df["x"].to_numpy()).astype(np.float32)
    Y = np.stack(df["y"].to_numpy()).astype(np.float32)
    meta = df[META_COLUMNS].reset_index(drop=True)
    return X, Y, meta


def load_shuffled_y(
    name: SplitName,
    seed: int = 42,
    dataset_path: Path = DATASET_PATH,
    splits_path: Path = SPLITS_PATH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Same as load_split but Y is permuted within the split. Anti-baseline control."""
    X, Y, meta = load_split(name, dataset_path=dataset_path, splits_path=splits_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    return X, Y[perm], meta
