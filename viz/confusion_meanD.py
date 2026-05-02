"""Confusion matrix heatmap: NT-v2 + meanD on the 5-way family task.

Headline-figure replacement for the unsupervised UMAP. Reads the saved
confusion JSON and renders a row-normalised 5x5 heatmap with raw counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SRC = DATA / "confusion_5way_nt_v2_meanD.json"
OUT = Path(__file__).resolve().parent / "figures" / "confusion_5way_nt_v2_meanD.png"

DISPLAY_ORDER = ["tf", "gpcr", "kinase", "ion", "immune"]
DISPLAY_LABELS = {
    "tf": "TF",
    "gpcr": "GPCR",
    "kinase": "Kinase",
    "ion": "Ion channel",
    "immune": "Immune receptor",
}


def main():
    with open(SRC) as f:
        payload = json.load(f)

    classes = payload["classes"]
    matrix = np.array(payload["matrix"], dtype=int)

    idx = [classes.index(c) for c in DISPLAY_ORDER]
    matrix = matrix[np.ix_(idx, idx)]
    labels = [DISPLAY_LABELS[c] for c in DISPLAY_ORDER]

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = matrix / np.maximum(row_totals, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(
        "5-way family classification — NT-v2 + meanD\n"
        "Test confusion matrix (row-normalised; raw counts in cells)",
        fontsize=11,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = matrix[i, j]
            frac = normalised[i, j]
            colour = "white" if frac > 0.55 else "black"
            ax.text(
                j, i,
                f"{count}\n({frac:.2f})",
                ha="center", va="center",
                fontsize=9, color=colour,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised fraction")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
