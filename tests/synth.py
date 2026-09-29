"""Tiny synthetic datasets for the probe-core guard tests.

Each dataset is a parquet with the real schema ({ensembl_id, x, y, symbol,
family, summary}) plus a splits file, so the CLIs run on it unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("gpcr", "immune", "ion", "kinase", "tf")


def _frame(ids, X, Y, fam) -> pd.DataFrame:
    return pd.DataFrame({
        "ensembl_id": ids,
        "x": [row.astype(np.float32) for row in X],
        "y": [row.astype(np.float32) for row in Y],
        "symbol": [f"SYM{i}" for i in range(len(ids))],
        "family": fam,
        "summary": ["" for _ in ids],
    })


def write_dataset(tmp: Path, *, sizes=(150, 60, 60), d=16, k=8, seed=0,
                  ridge_beta2: float | None = None, permute_test_labels: bool = False,
                  name: str = "synthetic") -> tuple[Path, Path]:
    """Write ``<name>.parquet`` and ``<name>_splits.json`` under ``tmp``.

    Classification signal: each family shifts the feature mean. With
    ``ridge_beta2`` the GenePT-like target is ``X @ B + noise`` with
    ``B ~ N(0, beta2)``, whose validation R^2 peaks at alpha = 1e4 for seed 0
    (600/300/300 genes, d=200, k=16).
    """
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    fam = np.array([FAMILIES[i % 5] for i in range(n)])
    if ridge_beta2 is None:
        centers = rng.standard_normal((5, d)) * 0.8
        X = rng.standard_normal((n, d)) + centers[[FAMILIES.index(f) for f in fam]]
        Y = 0.5 * X[:, :k] + 0.5 * rng.standard_normal((n, k))
    else:
        X = rng.standard_normal((n, d))
        B = rng.standard_normal((d, k)) * np.sqrt(ridge_beta2)
        Y = X @ B + rng.standard_normal((n, k))
    ids = np.array([f"ENSG{i:011d}" for i in range(n)])
    n_tr, n_va, _ = sizes
    split = {"train": ids[:n_tr].tolist(), "val": ids[n_tr:n_tr + n_va].tolist(),
             "test": ids[n_tr + n_va:].tolist()}
    if permute_test_labels:
        te = np.arange(n_tr + n_va, n)
        fam[te] = np.random.default_rng(seed + 99).permutation(fam[te])
    tmp.mkdir(parents=True, exist_ok=True)
    parquet = tmp / f"{name}.parquet"
    _frame(ids, X, Y, fam).to_parquet(parquet)
    splits = tmp / f"{name}_splits.json"
    splits.write_text(json.dumps(split))
    return parquet, splits


class LoggedSplits(dict):
    """A splits dict that logs which split names are read, in order."""

    def __init__(self, data, log):
        super().__init__(data)
        self._log = log

    def __getitem__(self, key):
        self._log.append(key)
        return super().__getitem__(key)


def redirect_splits(monkeypatch, splits_file: Path, log: list | None = None) -> None:
    """Make every ``load_split`` read ``splits_file`` whatever path it is given."""
    import splits.loader as loader

    def _read(_path=None):
        data = json.loads(Path(splits_file).read_text())
        return LoggedSplits(data, log) if log is not None else data

    monkeypatch.setattr(loader, "_load_splits_file", _read)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def call_main(module, argv: list[str]) -> None:
    import sys
    old = sys.argv
    sys.argv = [module.__name__ + ".py", *argv]
    try:
        module.main()
    finally:
        sys.argv = old
