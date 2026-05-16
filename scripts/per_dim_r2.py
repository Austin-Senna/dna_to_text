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

from splits import load_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ANALYSIS_FIG = REPO / "analysis" / "figures"
OUT_JSON = DATA / "per_dim_r2.json"
OUT_PNG = ANALYSIS_FIG / "per_dim_r2_distribution.png"

# Headline regression cells: (label, probe_npz, dataset_parquet)
CELLS = [
    ("CDS DNABERT-2 meanG",  "probe_dnabert2_meanG.npz",  "dataset_dnabert2_meanG.parquet"),
    ("CDS NT-v2 meanmean",   "probe_nt_v2_meanmean.npz",  "dataset_nt_v2_meanmean.parquet"),
    ("CDS HyenaDNA special", "probe_hyena_dna_specialmean.npz", "dataset_hyena_dna_specialmean.parquet"),
    ("CDS GENA-LM meanmean", "probe_gena_lm_meanmean.npz", "dataset_gena_lm_meanmean.parquet"),
    ("TSS DNABERT-2 meanmean", "probe_tss_dnabert2_meanmean.npz", "dataset_tss_dnabert2_meanmean.parquet"),
]


def per_dim_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    Y_mean = Y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Y_true - Y_mean) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)


def main() -> None:
    results: dict[str, dict] = {}

    for label, probe_name, dataset_name in CELLS:
        probe_path = DATA / probe_name
        dataset_path = DATA / dataset_name
        if not probe_path.exists() or not dataset_path.exists():
            print(f"  SKIP {label}: missing {probe_path.name} or {dataset_path.name}")
            continue

        print(f"=== {label} ===")
        probe = np.load(probe_path)
        W, b = probe["W"], probe["b"]
        alpha = float(probe["alpha"])

        X_te, Y_te, _ = load_split("test", dataset_path=dataset_path)
        Y_pred = X_te @ W + b
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
            "probe": probe_name,
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
    ax.set_title("Per-dimension R² histogram")
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
    ax.set_title("Rank-ordered per-dim R²")
    ax.grid(axis="both", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)

    fig.suptitle("GenePT regression: per-dimension R² across 1,536 dims")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
