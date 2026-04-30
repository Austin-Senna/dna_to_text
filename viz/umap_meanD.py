"""Single-panel UMAP: NT-v2 + meanD pooling, coloured by family.

The headline-pipeline visual. Anchors the "encoder carries family-
discriminative signal" claim.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT = Path(__file__).resolve().parent / "figures" / "umap_nt_v2_meanD.png"

FAMILY_PALETTE = {
    "tf":     "#1f77b4",
    "gpcr":   "#d62728",
    "kinase": "#2ca02c",
    "ion":    "#9467bd",
    "immune": "#ff7f0e",
}


def main():
    df = pd.read_parquet(DATA / "dataset_nt_v2_meanD.parquet")
    X = np.stack(df["x"].values).astype(np.float32)
    families = df["family"].values
    print(f"  computing UMAP on NT-v2 + meanD (n={len(df)}, dim={X.shape[1]})...")

    coords = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    for fam, colour in FAMILY_PALETTE.items():
        mask = families == fam
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=6, alpha=0.65, c=colour,
                   label=f"{fam} (n={int(mask.sum())})")
    ax.set_title(
        "NT-v2 + meanD pooling — coloured by functional family\n"
        "(headline-pipeline: tokenise with specials, mean-pool tokens per chunk,\n"
        " concat[first_chunk, last_chunk, mean_chunks] across chunks)",
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=9, markerscale=2.0, framealpha=0.85)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
