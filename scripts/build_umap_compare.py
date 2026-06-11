#!/usr/bin/env python3
"""Two paired-UMAP comparison figures for the Results section (no embedded
figure titles -- the LaTeX caption is the title; small per-panel labels stay,
since the caption cannot identify individual panels).

  umap_cds_vs_tss.png  -- NT-v2 CDS (meanG) beside NT-v2 TSS-window (meanmean):
        the family clusters present on coding sequence dissolve on the
        196,608 bp regulatory window (substrate collapse, sec 3.4).
  umap_cds_vs_esm.png  -- NT-v2 CDS (meanG) beside ESM-2 650M (translated CDS):
        the frozen DNA encoder versus the protein-LM upper bound (sec 3.6).

Run: uv run scripts/build_umap_compare.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures"

FAM_ORDER = ["tf", "gpcr", "kinase", "ion", "immune"]
FAM_DISP = {"tf": "TF", "gpcr": "GPCR", "kinase": "Kinase", "ion": "Ion channel",
            "immune": "Immune receptor"}
FAM_COLOR = {"tf": "#1f77b4", "gpcr": "#ff7f0e", "kinase": "#2ca02c",
             "ion": "#d62728", "immune": "#9467bd"}


def _coords(parquet: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA / parquet)
    X = np.stack(df["x"].values).astype(np.float32)
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       random_state=42).fit_transform(X)
    return coords, df["family"].values


def _panel(ax, coords, fams, label):
    for fam in FAM_ORDER:
        m = fams == fam
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6,
                   c=FAM_COLOR[fam], linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#bbbbbb")
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def _figure(left, right, fname):
    (cl, fl), llabel = left
    (cr, fr), rlabel = right
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    _panel(axes[0], cl, fl, llabel)
    _panel(axes[1], cr, fr, rlabel)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=FAM_COLOR[f], markeredgewidth=0,
                      label=FAM_DISP[f]) for f in FAM_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cds = (_coords("dataset_nt_v2_meanG.parquet"), "NT-v2 CDS")
    tss = (_coords("dataset_tss_nt_v2_meanmean.parquet"), "NT-v2 TSS window")
    esm = (_coords("dataset_esm2_650m.parquet"), "ESM-2 650M (protein)")
    _figure(cds, tss, "umap_cds_vs_tss.png")
    _figure(cds, esm, "umap_cds_vs_esm.png")


if __name__ == "__main__":
    main()
