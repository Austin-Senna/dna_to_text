"""Build deterministic balanced binary subsets of the full corpus.

Each subset:
  - downsamples the larger class to match the smaller class size (seed=42)
  - then computes a stratified 70/15/15 split on the binary label (seed=42)
  - and freezes both the gene selection AND the split into one JSON file.

Every probe + baseline + anti-baseline run reads the same JSON, so the
comparison across runs is on the same genes in the same splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "dataset.parquet"

SEED = 42

BinaryTask = Literal["tf_vs_gpcr", "tf_vs_kinase"]

BINARY_TASKS: dict[str, tuple[str, str]] = {
    "tf_vs_gpcr": ("gpcr", "tf"),
    "tf_vs_kinase": ("kinase", "tf"),
}


def build_binary_subset(
    df: pd.DataFrame,
    task: BinaryTask,
    seed: int = SEED,
) -> dict:
    """Downsample the larger family to match the smaller, then make a stratified split.

    Returns a dict ready for json.dump (no numpy types).
    """
    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")
    pos_fam, neg_fam = BINARY_TASKS[task]

    pos = df[df["family"] == pos_fam].sort_values("ensembl_id").reset_index(drop=True)
    neg = df[df["family"] == neg_fam].sort_values("ensembl_id").reset_index(drop=True)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"empty family for task {task}: pos={len(pos)} neg={len(neg)}")

    n = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(len(pos), size=n, replace=False)
    neg_idx = rng.choice(len(neg), size=n, replace=False)
    pos_ids = sorted(pos.iloc[pos_idx]["ensembl_id"].tolist())
    neg_ids = sorted(neg.iloc[neg_idx]["ensembl_id"].tolist())

    all_ids = pos_ids + neg_ids
    labels = [1] * len(pos_ids) + [0] * len(neg_ids)

    train_ids, rest_ids, train_y, rest_y = train_test_split(
        all_ids, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, random_state=seed, stratify=rest_y
    )

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(all_ids)

    return {
        "task": task,
        "seed": seed,
        "positive_label": pos_fam,
        "negative_label": neg_fam,
        "n_per_class": n,
        "positive_ensembl_ids": pos_ids,
        "negative_ensembl_ids": neg_ids,
        "split": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }


def write_binary_subset_json(
    task: BinaryTask,
    out_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET,
    seed: int = SEED,
) -> dict:
    df = pd.read_parquet(dataset_path)
    payload = build_binary_subset(df, task, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
