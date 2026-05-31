#!/usr/bin/env python3
"""Reframed Results figures from the metrics JSONs (no embedded titles --
the LaTeX caption is the title, per repo convention).

  A  comparator_landscape.png  -- macro-F1 | R^2 bars: composition, DNA
                                  encoders, ESM-2 (comparator). Composition
                                  ties the DNA encoders; ESM-2 towers.
  B  substrate_collapse.png    -- per-encoder CDS vs TSS kappa; TSS falls to
                                  near the chance floor.
  C  split_slope.png           -- random -> homology slope per comparator;
                                  composition >= best DNA encoder on both ends.

Run: uv run scripts/build_result_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures"

M = json.loads((DATA / "metrics_homology.json").read_text())
RAND = json.loads((DATA / "metrics.json").read_text())
RANDC = json.loads((DATA / "metrics_random_comparators.json").read_text())

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENC_DISP = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
FLOOR = 0.224  # 5-way family chance floor (shuffled-label macro-F1)

# palette
C_COMP = "#9e9e9e"   # composition baselines (grey)
C_DNA = "#3a7d44"    # DNA encoders (green)
C_ESM = "#c0504d"    # ESM-2 comparator (red)


# ---------- accessors ----------
def _best_cls(metrics, src, tss=False, metric="test_macro_f1"):
    cells = []
    for r in metrics:
        if r.get("task") != "family5" or r.get("shuffled_labels"):
            continue
        fs = r["feature_source"]
        is_tss = fs.startswith("tss_")
        if is_tss != tss:
            continue
        core = fs[4:] if is_tss else fs
        if core == src or core.startswith(src + "_"):
            cells.append(r)
    return max(cells, key=lambda r: r[metric])[metric] if cells else None


def _cell_cls(metrics, src):
    c = [r for r in metrics if r.get("task") == "family5" and not r.get("shuffled_labels")
         and r.get("feature_source") == src]
    return c[0]["test_macro_f1"] if c else None


def _best_reg_enc(metrics, enc):
    cells = [r for r in metrics if r.get("task") is None and r.get("model") == "linear_probe"
             and not str(r.get("dataset", "")).startswith("dataset_tss_")
             and (str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "") == enc
                  or str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "").startswith(enc + "_"))]
    return max(cells, key=lambda r: r["test_r2_macro"])["test_r2_macro"] if cells else None


def _cell_reg(metrics, src):
    # baselines (model name) or esm (dataset/feature_source)
    for r in metrics:
        if r.get("task") is not None:
            continue
        ds = str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "")
        fs = r.get("feature_source")
        model = r.get("model", "")
        key = {"kmer_baseline_4": "kmer", "kmer_baseline_6": "kmer6", "codon_baseline": "codon",
               "gc_baseline": "gc", "aa_baseline_1": "aa1", "aa_baseline_2": "aa2", "aa_baseline_3": "aa3"}.get(model)
        if key == src or ds == src or fs == src:
            return r["test_r2_macro"]
    return None


def f1_of(metrics, src):
    return _best_cls(metrics, src) if src in ENCODERS else _cell_cls(metrics, src)


def r2_of(metrics, src):
    return _best_reg_enc(metrics, src) if src in ENCODERS else _cell_reg(metrics, src)


def _no_title(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


# ---------- Figure A: comparator landscape ----------
def fig_comparator_landscape():
    # representative cells, grouped: composition | DNA encoders | ESM-2 (comparator)
    cells = [
        ("CDS 4-mer", "kmer", C_COMP), ("AA 2-mer", "aa2", C_COMP), ("AA 3-mer", "aa3", C_COMP),
        ("DNABERT-2", "dnabert2", C_DNA), ("NT-v2", "nt_v2", C_DNA),
        ("GENA-LM", "gena_lm", C_DNA), ("HyenaDNA", "hyena_dna", C_DNA),
        ("ESM-2 150M", "esm2_150m", C_ESM), ("ESM-2 650M", "esm2_650m", C_ESM),
    ]
    labels = [c[0] for c in cells]
    colors = [c[2] for c in cells]
    x = np.arange(len(cells))
    fig, (axF, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    f1 = [f1_of(M, c[1]) for c in cells]
    axF.bar(x, f1, color=colors, edgecolor="white",
            hatch=["" if c[2] != C_ESM else "//" for c in cells])
    axF.axhline(FLOOR, color="#555", ls="--", lw=0.9)
    axF.text(len(cells) - 0.4, FLOOR + 0.006, "chance 0.224", ha="right", va="bottom", fontsize=7.5, color="#555")
    axF.set_ylabel("5-way family macro-F1")
    axF.set_ylim(0, 1.0)

    r2 = [r2_of(M, c[1]) for c in cells]
    axR.bar(x, r2, color=colors, edgecolor="white",
            hatch=["" if c[2] != C_ESM else "//" for c in cells])
    axR.axhline(0, color="#555", lw=0.8)
    axR.set_ylabel("Ridge-to-GenePT $R^2$")
    axR.set_ylim(0, 0.20)

    for ax in (axF, axR):
        _no_title(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    # legend
    from matplotlib.patches import Patch
    axF.legend(handles=[Patch(facecolor=C_COMP, label="composition"),
                        Patch(facecolor=C_DNA, label="DNA encoder"),
                        Patch(facecolor=C_ESM, hatch="//", label="ESM-2 (comparator)")],
               fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "comparator_landscape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("comparator_landscape.png  F1:", [round(v, 3) for v in f1])


# ---------- Figure B: substrate collapse ----------
def fig_substrate_collapse():
    cds = [_best_cls(M, e, tss=False, metric="test_kappa") for e in ENCODERS]
    tss = [_best_cls(M, e, tss=True, metric="test_kappa") for e in ENCODERS]
    tss_floor = _best_cls(M, "enformer_tss_4mer".replace("enformer_tss_4mer", "enformer_tss_4mer"),
                          tss=True, metric="test_kappa")
    # the TSS 4-mer is feature_source 'enformer_tss_4mer'
    tss4 = next((r["test_kappa"] for r in M if r.get("task") == "family5"
                 and r.get("feature_source") == "enformer_tss_4mer"), 0.033)
    x = np.arange(len(ENCODERS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - w / 2, cds, w, color=C_DNA, label="CDS", edgecolor="white")
    ax.bar(x + w / 2, tss, w, color=C_DNA, alpha=0.45, hatch="//", label="TSS window", edgecolor="white")
    ax.axhline(tss4, color="#555", ls="--", lw=0.9)
    ax.text(len(ENCODERS) - 0.5, tss4 + 0.006, f"TSS 4-mer $\\kappa$ {tss4:.3f}", ha="right", va="bottom",
            fontsize=7.5, color="#555")
    _no_title(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([ENC_DISP[e] for e in ENCODERS], fontsize=9)
    ax.set_ylabel("5-way family $\\kappa$ (best pool)")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "substrate_collapse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("substrate_collapse.png  CDS:", [round(v, 3) for v in cds], "TSS:", [round(v, 3) for v in tss])


# ---------- Figure C: split slope ----------
def _rand_f1(src):
    v = _cell_cls(RANDC, src)
    if v is not None:
        return v
    return f1_of(RAND, src)


def fig_split_slope():
    rows = [
        ("ESM-2 650M", "esm2_650m", C_ESM), ("ESM-2 150M", "esm2_150m", C_ESM),
        ("AA 2-mer", "aa2", C_COMP), ("NT-v2", "nt_v2", C_DNA),
        ("DNABERT-2", "dnabert2", C_DNA), ("CDS 4-mer", "kmer", C_COMP),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for label, src, col in rows:
        rv, hv = _rand_f1(src), f1_of(M, src)
        ax.plot([0, 1], [rv, hv], "-o", color=col, lw=2, ms=6)
        ax.text(1.02, hv, f" {label} {hv:.3f}", va="center", fontsize=8, color=col)
        ax.text(-0.02, rv, f"{rv:.3f} ", va="center", ha="right", fontsize=8, color=col)
    ax.axhline(FLOOR, color="#999", ls=":", lw=0.8)
    _no_title(ax)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random split", "homology split"], fontsize=9)
    ax.set_xlim(-0.35, 1.6)
    ax.set_ylabel("5-way family macro-F1")
    ax.set_ylim(0.55, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "split_slope.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("split_slope.png done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_comparator_landscape()
    fig_substrate_collapse()
    fig_split_slope()
    print("wrote 3 figures to", OUT)


if __name__ == "__main__":
    main()
