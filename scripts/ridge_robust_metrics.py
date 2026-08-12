"""Rotation-invariant Ridge->GenePT metrics for the R2-metric rebuttal (MLCB R2-W4).

Reviewer 2 noted macro-averaged per-dimension R^2 is coordinate-dependent (equal weight
to all 1,536 GenePT dims) and might miss shared structure spread across dim-combinations.
For each cell this refits Ridge at the recorded alpha on train+val (same recipe as
per_dim_r2.py), predicts on held-out test, and reports three views:
  - macro_r2   : uniform_average per-dim R^2 (the reported headline)
  - pooled_r2  : variance_weighted = 1 - sum(SS_res)/sum(SS_tot)   (rotation-robust)
  - retrieval  : cosine nearest-neighbour retrieval of the true gene in GenePT space
                 (top1/top5/top10, median rank) -- coordinate-free and combination-aware.
Writes data/ridge_robust.json (consumed by build_paper_tables.build_ridge_robust).

Run: uv run scripts/ridge_robust_metrics.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

# (display label, dataset parquet). ESM-2 upper bound -> best DNA -> weakest -> TSS.
CELLS = [
    ("ESM-2 650M (upper bound)", "dataset_esm2_650m.parquet"),
    ("ESM-2 150M", "dataset_esm2_150m.parquet"),
    ("DNABERT-2 meanD (best DNA)", "dataset_dnabert2_meanD.parquet"),
    ("NT-v2 meanG", "dataset_nt_v2_meanG.parquet"),
    ("HyenaDNA meanG", "dataset_hyena_dna_meanG.parquet"),
    ("GENA-LM meanG (weakest)", "dataset_gena_lm_meanG.parquet"),
    ("TSS DNABERT-2 meanmean", "dataset_tss_dnabert2_meanmean.parquet"),
]


def alpha_for(ds: str, metrics: list[dict]) -> float:
    """Recorded validation-selected Ridge alpha for a dataset (default 10.0)."""
    for run in metrics:
        if run.get("model") == "linear_probe" and run.get("dataset") == ds:
            return float(run["alpha"])
    return 10.0


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted (pooled) R^2 -- invariant to rotations of the target basis."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return 1.0 - ss_res / ss_tot


def retrieval(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Cosine nearest-neighbour retrieval of each gene's true GenePT vector."""
    a = y_pred / np.clip(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-12, None)
    b = y_true / np.clip(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-12, None)
    sim = a @ b.T
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
    return {
        "n_test": int(n),
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "chance_top1": 1.0 / n,
        "chance_top5": 5.0 / n,
    }


def run_cell(label: str, ds: str, metrics: list[dict], shuffle: bool = False) -> dict | None:
    path = DATA / ds
    if not path.exists():
        return None
    alpha = alpha_for(ds, metrics)
    x_tr, y_tr, _ = load_split("train", dataset_path=path)
    x_va, y_va, _ = load_split("val", dataset_path=path)
    x_te, y_te, _ = load_split("test", dataset_path=path)
    x_fit, y_fit = np.vstack([x_tr, x_va]), np.vstack([y_tr, y_va])
    if shuffle:
        rng = np.random.default_rng(0)
        y_fit = y_fit[rng.permutation(len(y_fit))]
    model = Ridge(alpha=alpha).fit(x_fit, y_fit)
    y_pred = model.predict(x_te)
    out = {
        "label": label,
        "dataset": ds,
        "alpha": alpha,
        "macro_r2": float(r2_score(y_te, y_pred, multioutput="uniform_average")),
        "pooled_r2": float(pooled_r2(y_te, y_pred)),
    }
    out.update(retrieval(y_pred, y_te))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "ridge_robust.json")
    args = ap.parse_args()

    metrics = json.loads((DATA / "metrics_homology.json").read_text())
    rows = [r for label, ds in CELLS if (r := run_cell(label, ds, metrics))]
    ctrl = run_cell("SHUFFLED-TARGET control", "dataset_dnabert2_meanD.parquet",
                    metrics, shuffle=True)
    if ctrl:
        rows.append(ctrl)

    hdr = (f'{"cell":28s} {"macroR2":>8s} {"pooledR2":>8s} '
           f'{"top1%":>7s} {"top5%":>7s} {"top10%":>7s} {"medRank":>8s}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f'{r["label"][:27]:28s} {r["macro_r2"]:8.4f} {r["pooled_r2"]:8.4f} '
              f'{100*r["top1"]:7.2f} {100*r["top5"]:7.2f} {100*r["top10"]:7.2f} '
              f'{r["median_rank"]:8.0f}')
    print(f'\nn_test = {rows[0]["n_test"]}   chance top5 = {100*rows[0]["chance_top5"]:.3f}%'
          f'   chance median rank ~ {rows[0]["n_test"]//2}')
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
