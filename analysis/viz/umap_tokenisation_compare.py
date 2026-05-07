"""Two-panel UMAP: DNABERT-2 mean-pool BEFORE vs AFTER the tokenisation fix.

Before: dataset.parquet (Phase 1-3, add_special_tokens=False, no [CLS]/[SEP]
        wrapping per chunk).
After:  dataset_dnabert2_meanmean.parquet (Phase 4b, with-specials, identical
        mean->mean pooling otherwise).

Same gene set, same family colouring. Visualises the Phase 4b "tokenisation
surprise" — DNABERT-2 should show visibly tighter family clusters in the
After panel even though the pooling pattern is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
OUT = Path(__file__).resolve().parent / "figures" / "umap_dnabert2_tokenisation_compare.png"

FAMILY_PALETTE = {
    "tf":     "#1f77b4",
    "gpcr":   "#d62728",
    "kinase": "#2ca02c",
    "ion":    "#9467bd",
    "immune": "#ff7f0e",
}


def _embed(X: np.ndarray, seed: int = 42) -> np.ndarray:
    return umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed
    ).fit_transform(X)


def _scatter(ax, coords, families, title):
    for fam, colour in FAMILY_PALETTE.items():
        mask = families == fam
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=4, alpha=0.6, c=colour, label=f"{fam} (n={int(mask.sum())})")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def main():
    before = pd.read_parquet(DATA / "dataset.parquet").sort_values("ensembl_id").reset_index(drop=True)
    after = pd.read_parquet(DATA / "dataset_dnabert2_meanmean.parquet").sort_values("ensembl_id").reset_index(drop=True)
    assert (before["ensembl_id"].values == after["ensembl_id"].values).all(), \
        "gene order mismatch between before/after parquets"
    families = before["family"].values

    print(f"  computing UMAP for BEFORE (no specials, dim={np.stack(before['x'].values).shape[1]})...")
    coords_before = _embed(np.stack(before["x"].values).astype(np.float32))
    print(f"  computing UMAP for AFTER (with specials, dim={np.stack(after['x'].values).shape[1]})...")
    coords_after = _embed(np.stack(after["x"].values).astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], coords_before, families,
             "DNABERT-2  ·  mean→mean  ·  add_special_tokens=False\n(Phase 1–3 pipeline)")
    _scatter(axes[1], coords_after, families,
             "DNABERT-2  ·  mean→mean  ·  add_special_tokens=True\n(Phase 4b re-extraction — same model, same pooling)")
    axes[1].legend(loc="upper right", fontsize=8, markerscale=2.0, framealpha=0.85)
    fig.suptitle("Tokenisation surprise: identical pooling, identical model — only the boundary tokens differ", fontsize=11)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
