"""Build a frozen stratified 70/15/15 split of dataset.parquet and write data/splits.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
STRATIFY_COL = "family"


def build_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    ids = df["ensembl_id"].tolist()
    fams = df[STRATIFY_COL].tolist()

    train_ids, rest_ids, _, rest_fams = train_test_split(
        ids, fams, test_size=0.30, random_state=seed, stratify=fams
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, random_state=seed, stratify=rest_fams
    )

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(ids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_splits_json(
    dataset_path: str | Path,
    out_path: str | Path,
    seed: int = SEED,
) -> dict:
    df = pd.read_parquet(dataset_path)
    parts = build_splits(df, seed=seed)
    payload = {
        **parts,
        "seed": seed,
        "stratify": STRATIFY_COL,
        "source": str(Path(dataset_path).name),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
