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


def resolve_dataset_path(dataset_path: Path | None = None) -> Path:
    """Resolve the parquet to read.

    An explicit path is honoured as-is. Otherwise the canonical
    ``data/dataset.parquet`` is used when present; if it is absent (some
    checkouts only materialise the per-encoder pooling variants) we fall back
    to any present ``dataset_*_meanmean.parquet``. All encoder parquets share
    the same ``{ensembl_id, x, y, symbol, family, summary}`` schema and an
    identical GenePT target ``y`` / metadata, so the fallback is correct for
    callers that only need ``y`` and ``meta`` (e.g. the compositional
    baselines, which compute their own ``x``).
    """
    if dataset_path is not None:
        return Path(dataset_path)
    if DATASET_PATH.exists():
        return DATASET_PATH
    for cand in sorted(DATA_DIR.glob("dataset_*_meanmean.parquet")):
        return cand
    return DATASET_PATH  # nothing found; let the reader raise an informative error


def _load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(resolve_dataset_path(dataset_path))


def _load_splits_file(splits_path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(splits_path).read_text())


def load_split(
    name: SplitName,
    dataset_path: Path | None = None,
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
    dataset_path: Path | None = None,
    splits_path: Path = SPLITS_PATH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Same as load_split but Y is permuted within the split. Anti-baseline control."""
    X, Y, meta = load_split(name, dataset_path=dataset_path, splits_path=splits_path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    return X, Y[perm], meta
