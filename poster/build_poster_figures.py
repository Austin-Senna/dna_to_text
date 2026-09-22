#!/usr/bin/env python3
"""Poster-styled figures (vector PDF) for the MINA URF symposium poster.

Reuses the paper's metric accessors (scripts/build_result_figures.py) and the
UMAP embedding computation (scripts/build_umap_compare.py) so every printed
value matches the paper exactly, then re-renders larger, thicker, print-legible
panels into poster/figures/ as PDF. The paper's own figure scripts are left
untouched; only styling, colour, and output path/format differ here.

Colour-vision-deficiency safety: the bar charts use a role palette with no
red/green (grey composition, blue DNA encoder, orange protein-LM ceiling), and
the UMAP uses the Okabe-Ito family palette PLUS a distinct marker shape per
family, so families stay separable under red-green CVD on dense overlapping
points.

Run: uv run poster/build_poster_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_result_figures import (  # noqa: E402  (sys.path set just above)
    CELLS, C_COMP, C_DNA, C_ESM, ENCODERS, ENC_DISP, ENFH, FLOOR, M,
    _aa_best, _best_cls, _rand_f1, f1_of,
)
from build_umap_compare import (  # noqa: E402
    FAM_DISP, FAM_ORDER, _coords,
)

OUT = ROOT / "poster" / "figures"

plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 20,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 18,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "pdf.fonttype": 42,
})

# Colour-blind-safe role palette for the bar charts (no red/green): grey =
# composition, blue = frozen DNA encoder, orange = protein-LM ceiling. Remap
# the imported paper colours onto these roles.
PC_COMP, PC_DNA, PC_ESM = "#8A8A8A", "#3B6FB0", "#E07B39"
_ROLE = {C_COMP: PC_COMP, C_DNA: PC_DNA, C_ESM: PC_ESM}
_CHANCE = "#444444"

# Okabe-Ito family palette + per-family marker shape for the UMAP (shape is the
# redundancy that survives red-green CVD on dense overlapping points).
FAM_OKABE = {"tf": "#0072B2", "gpcr": "#E69F00", "kinase": "#009E73",
             "ion": "#D55E00", "immune": "#CC79A7"}
FAM_MARK = {"tf": "o", "gpcr": "^", "kinase": "s", "ion": "D", "immune": "P"}


# ---------- shared poster helpers ----------
def _grid(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.9)


def _bar_labels(ax, xs, vals, fmt="{:.2f}", dy=0.015, fs=17):
    top = ax.get_ylim()[1]
    for x, v in zip(xs, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(x, min(v + dy, top - dy), fmt.format(v), ha="center",
                va="bottom", fontsize=fs, color="#222")


def _chance(ax, x=0.0):
    ax.axhline(FLOOR, color="#555", ls="--", lw=1.5)
    ax.text(x, FLOOR + 0.02, f"chance {FLOOR:.3f}", fontsize=16, color=_CHANCE)


def _legend(ax, handles, loc):
    ax.legend(handles=handles, loc=loc, frameon=True, framealpha=0.95,
              edgecolor="none")


def _f1_where(fs):
    return next(r["test_macro_f1"] for r in M
                if r.get("task") == "family5" and r["feature_source"] == fs)


# ---------- figures ----------
def fig_comparator_f1():
    """The rigorous panel: 5-way family macro-F1, homology split."""
    labels = ["AA 2-mer" if c[0] == "AA comp." else c[0] for c in CELLS]
    colors = [_ROLE[c[2]] for c in CELLS]
    vals = [_aa_best(f1_of) if c[1] == "aa_best" else f1_of(M, c[1]) for c in CELLS]
    x = np.arange(len(CELLS))
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.6,
           hatch=["//" if c[2] == C_ESM else "" for c in CELLS])
    ax.set_ylim(0, 1.15)
    _bar_labels(ax, x, vals, dy=0.015, fs=17)
    _chance(ax, x=0.1)
    _grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("5-way family macro-F1\n(higher is better)")
    _legend(ax, [Patch(facecolor=PC_COMP, label="composition baseline"),
                 Patch(facecolor=PC_DNA, label="frozen DNA encoder"),
                 Patch(facecolor=PC_ESM, hatch="//", label="ESM-2 (protein LM, ceiling)")],
            "upper left")
    fig.tight_layout()
    fig.savefig(OUT / "comparator_f1.pdf")
    plt.close(fig)
    print("comparator_f1", dict(zip(labels, [round(v, 3) for v in vals])))


def fig_substrate_collapse():
    """CDS vs 196,608 bp TSS window: 5-way family macro-F1 (single panel)."""
    cats = ["CDS 4-mer"] + [ENC_DISP[e] for e in ENCODERS] + ["Enformer"]
    cols = [PC_COMP] + [PC_DNA] * 4 + [PC_ESM]
    x = np.arange(len(cats))
    w = 0.38

    enf_cls = max((r for r in ENFH if r.get("task") == "family5"),
                  key=lambda r: r["test_macro_f1"])
    cds_f1 = [_f1_where("kmer")] + [_best_cls(M, e, False, "test_macro_f1") for e in ENCODERS] + [np.nan]
    tss_f1 = [_f1_where("enformer_tss_4mer")] + [_best_cls(M, e, True, "test_macro_f1") for e in ENCODERS] + [enf_cls["test_macro_f1"]]

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.bar(x - w / 2, cds_f1, w, color=cols, edgecolor="white", linewidth=1.5)
    ax.bar(x + w / 2, tss_f1, w, color=cols, alpha=0.42, hatch="//", edgecolor="white", linewidth=1.5)
    _chance(ax, x=-0.15)
    _grid(ax)
    ax.set_ylabel("5-way family macro-F1\n(higher is better)")
    ax.set_ylim(0, 1.15)
    _bar_labels(ax, x - w / 2, cds_f1, dy=0.015, fs=15)
    _bar_labels(ax, x + w / 2, tss_f1, dy=0.015, fs=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    _legend(ax, [Patch(facecolor="#777", label="coding sequence (CDS)"),
                 Patch(facecolor="#777", alpha=0.42, hatch="//", label="TSS regulatory window"),
                 Patch(facecolor=PC_ESM, label="supervised comparator (Enformer)")],
            "upper right")
    fig.tight_layout()
    fig.savefig(OUT / "substrate_collapse.pdf")
    plt.close(fig)
    print("substrate F1 CDS:", [round(v, 3) for v in cds_f1], "TSS:", [round(v, 3) for v in tss_f1])


def fig_split_bars():
    """Random vs homology-aware split, per comparator: 5-way family macro-F1."""
    cells = [("CDS 4-mer", "kmer", PC_COMP), ("AA 2-mer", "aa2", PC_COMP), ("NT-v2", "nt_v2", PC_DNA),
             ("DNABERT-2", "dnabert2", PC_DNA), ("ESM-2 650M", "esm2_650m", PC_ESM)]
    cols = [c for *_, c in cells]
    labels = [c[0] for c in cells]
    x = np.arange(len(cells))
    w = 0.38
    f1_rand = [_rand_f1(s) for _, s, _ in cells]
    f1_hom = [f1_of(M, s) for _, s, _ in cells]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.bar(x - w / 2, f1_rand, w, color=cols, alpha=0.5, edgecolor="white", linewidth=1.5)
    ax.bar(x + w / 2, f1_hom, w, color=cols, edgecolor="white", linewidth=1.5)
    _chance(ax, x=-0.15)
    _grid(ax)
    ax.set_ylabel("5-way family macro-F1\n(higher is better)")
    ax.set_ylim(0, 1.15)
    _bar_labels(ax, x - w / 2, f1_rand, dy=0.016, fs=16)
    _bar_labels(ax, x + w / 2, f1_hom, dy=0.016, fs=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _legend(ax, [Patch(facecolor="#777", alpha=0.5, label="random split (leaks paralogs)"),
                 Patch(facecolor="#777", label="homology-aware split")],
            "upper left")
    fig.tight_layout()
    fig.savefig(OUT / "split_bars.pdf")
    plt.close(fig)
    print("split_bars F1 random:", [round(v, 3) for v in f1_rand], "homology:", [round(v, 3) for v in f1_hom])


def _umap_panel(ax, coords, fams, label):
    for fam in FAM_ORDER:
        m = fams == fam
        ax.scatter(coords[m, 0], coords[m, 1], s=15, alpha=0.7, c=FAM_OKABE[fam],
                   marker=FAM_MARK[fam], linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="top", ha="left")


def fig_umap_cds_vs_tss():
    """Hero: NT-v2 embeddings cluster by family on CDS, collapse on the TSS window."""
    cds_c, cds_f = _coords("dataset_nt_v2_meanG.parquet")
    tss_c, tss_f = _coords("dataset_tss_nt_v2_meanmean.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9))
    _umap_panel(axes[0], cds_c, cds_f, "Coding sequence (CDS)")
    _umap_panel(axes[1], tss_c, tss_f, "TSS regulatory window")
    handles = [Line2D([0], [0], marker=FAM_MARK[f], linestyle="", markersize=15,
                      markerfacecolor=FAM_OKABE[f], markeredgewidth=0,
                      label=FAM_DISP[f]) for f in FAM_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=17, columnspacing=1.3, handletextpad=0.35,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "umap_cds_vs_tss.pdf", dpi=300)
    plt.close(fig)
    print("wrote umap_cds_vs_tss.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_comparator_f1()
    fig_substrate_collapse()
    fig_split_bars()
    fig_umap_cds_vs_tss()
    print("wrote poster figures to", OUT)


if __name__ == "__main__":
    main()
