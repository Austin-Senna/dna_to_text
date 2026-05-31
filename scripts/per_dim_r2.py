"""Per-dimension R² distribution for GenePT regression probes (A·20).

For each headline regression cell, loads the saved Ridge probe weights
from data/probe_*.npz, predicts on the held-out test split, and
computes per-output-dim R² across the 1,536 GenePT dimensions. Outputs:

  - data/per_dim_r2.json   (rank-ordered per-dim R² per cell, summary stats)
  - analysis/figures/per_dim_r2_distribution.png   (histogram + cumulative)

Goal: distinguish "modest macro-R² because a few dims are very well
predicted (e.g., text-length proxies) and the rest are flat" from
"modest macro-R² because every dim is modestly predicted."
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge

from splits import load_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ANALYSIS_FIG = REPO / "analysis" / "figures"
OUT_JSON = DATA / "per_dim_r2.json"
OUT_PNG = ANALYSIS_FIG / "per_dim_r2_distribution.png"

# Homology-split headline regression cells: (label, dataset_parquet). The Ridge
# probe is re-fit on the homology train+val (alpha read from
# metrics_homology.json), not loaded from the random-split npz caches.
CELLS = [
    ("CDS DNABERT-2 meanD",    "dataset_dnabert2_meanD.parquet"),
    ("CDS NT-v2 meanG",        "dataset_nt_v2_meanG.parquet"),
    ("CDS HyenaDNA meanG",     "dataset_hyena_dna_meanG.parquet"),
    ("CDS GENA-LM meanG",      "dataset_gena_lm_meanG.parquet"),
    ("TSS DNABERT-2 meanmean", "dataset_tss_dnabert2_meanmean.parquet"),
]

_METRICS_HOMOLOGY = json.loads((DATA / "metrics_homology.json").read_text())


def _alpha_for(dataset_name: str) -> float:
    for run in _METRICS_HOMOLOGY:
        if run.get("model") == "linear_probe" and run.get("dataset") == dataset_name:
            return float(run["alpha"])
    return 10.0


def per_dim_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    Y_mean = Y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Y_true - Y_mean) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)


def main() -> None:
    results: dict[str, dict] = {}

    for label, dataset_name in CELLS:
        dataset_path = DATA / dataset_name
        if not dataset_path.exists():
            print(f"  SKIP {label}: missing {dataset_path.name}")
            continue

        print(f"=== {label} ===")
        alpha = _alpha_for(dataset_name)
        X_tr, Y_tr, _ = load_split("train", dataset_path=dataset_path)
        X_va, Y_va, _ = load_split("val", dataset_path=dataset_path)
        X_te, Y_te, _ = load_split("test", dataset_path=dataset_path)
        model = Ridge(alpha=alpha).fit(np.vstack([X_tr, X_va]), np.vstack([Y_tr, Y_va]))
        Y_pred = model.predict(X_te)
        r2 = per_dim_r2(Y_te, Y_pred)

        r2_sorted = np.sort(r2)[::-1]
        macro_r2 = float(r2.mean())
        median_r2 = float(np.median(r2))
        n_pos = int((r2 > 0).sum())
        n_high = int((r2 > 0.5).sum())
        cum50 = int(np.argmax(np.cumsum(r2_sorted) >= 0.5 * r2.sum()) + 1) if r2.sum() > 0 else -1

        print(f"  alpha           = {alpha}")
        print(f"  macro_r2 (mean) = {macro_r2:.4f}")
        print(f"  median_r2       = {median_r2:.4f}")
        print(f"  dims with R²>0  = {n_pos} / {len(r2)}")
        print(f"  dims with R²>0.5= {n_high}")
        print(f"  top dim R²      = {float(r2_sorted[0]):.4f}")
        print(f"  10th-pct dim R² = {float(np.percentile(r2, 10)):.4f}")
        print(f"  90th-pct dim R² = {float(np.percentile(r2, 90)):.4f}")
        print(f"  dims to reach 50% of summed R² = {cum50} / {len(r2)}")

        results[label] = {
            "dataset": dataset_name,
            "alpha": alpha,
            "n_dims": int(len(r2)),
            "macro_r2": macro_r2,
            "median_r2": median_r2,
            "n_dims_positive_r2": n_pos,
            "n_dims_r2_gt_0p5": n_high,
            "top_dim_r2": float(r2_sorted[0]),
            "p10_dim_r2": float(np.percentile(r2, 10)),
            "p90_dim_r2": float(np.percentile(r2, 90)),
            "dims_to_50pct_summed_r2": cum50,
            "r2_per_dim_sorted_desc": r2_sorted.astype(float).tolist(),
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    if not results:
        print("no cells, skipping plot")
        return

    ANALYSIS_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = plt.get_cmap("tab10")

    ax = axes[0]
    bins = np.linspace(-0.2, 0.6, 41)
    for i, (label, res) in enumerate(results.items()):
        r2 = np.asarray(res["r2_per_dim_sorted_desc"])
        ax.hist(r2, bins=bins, histtype="step", linewidth=1.8, color=colors(i), label=label)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("per-dim test R²")
    ax.set_ylabel("count of GenePT dims (of 1,536)")
    # [no-title convention] ax.set_title("Per-dimension R² histogram")
    ax.grid(axis="both", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for i, (label, res) in enumerate(results.items()):
        r2 = np.asarray(res["r2_per_dim_sorted_desc"])
        ax.plot(np.arange(1, len(r2) + 1), r2, color=colors(i), label=label, linewidth=1.5)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("dim rank (best → worst)")
    ax.set_ylabel("per-dim test R²")
    # [no-title convention] ax.set_title("Rank-ordered per-dim R²")
    ax.grid(axis="both", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)

    # [no-title convention] fig.suptitle("GenePT regression: per-dimension R² across 1,536 dims")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
