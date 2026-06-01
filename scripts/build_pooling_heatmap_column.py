#!/usr/bin/env python3
"""Regenerate the §3.3 pooling heatmap (encoder x pooling, macro-F1) for the paper.

Reproduces ``paper/figures/pooling_heatmap_family5_column.png`` from the
homology-split metrics. Per-cell value labels use an adaptive text colour
(white on dark cells, black on light) keyed to each cell's luminance under the
colormap, so the numbers stay legible on the dark NT-v2 row.

    uv run scripts/build_pooling_heatmap_column.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures" / "pooling_heatmap_family5_column.png"

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENC_DISPLAY = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
POOLS = ["meanmean", "specialmean", "meanD", "meanG", "maxmean", "clsmean"]
CMAP = "YlGnBu"


def main() -> None:
    metrics = json.loads((DATA / "metrics_homology.json").read_text())
    cell: dict[tuple[str, str], float] = {}
    for r in metrics:
        if r.get("task") != "family5" or r.get("shuffled_labels"):
            continue
        fs = str(r.get("feature_source", ""))
        for enc in ENCODERS:
            if fs.startswith(enc + "_"):
                pool = fs[len(enc) + 1 :]
                if pool in POOLS:
                    cell[(enc, pool)] = float(r["test_macro_f1"])

    values = np.full((len(ENCODERS), len(POOLS)), np.nan)
    for i, enc in enumerate(ENCODERS):
        for j, pool in enumerate(POOLS):
            if (enc, pool) in cell:
                values[i, j] = cell[(enc, pool)]

    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(POOLS)))
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_xticklabels(POOLS, rotation=35, ha="right", fontfamily="monospace")
    ax.set_yticklabels([ENC_DISPLAY[e] for e in ENCODERS])
    for i in range(len(ENCODERS)):
        for j in range(len(POOLS)):
            v = values[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(v))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            colour = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=colour)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("macro-F1")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
