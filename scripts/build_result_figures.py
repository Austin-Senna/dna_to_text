#!/usr/bin/env python3
"""Reframed Results figures from the metrics JSONs (no embedded titles --
the LaTeX caption is the title, per repo convention).

  comparator_f1.png / comparator_r2.png  -- composition, DNA encoders, ESM-2
        (comparator) on macro-F1 (sec 3.1) and GenePT R^2 (sec 3.2).
  substrate_collapse.png  -- CDS vs TSS kappa for 4-mer + DNA encoders, plus
        the supervised Enformer TSS comparator (sec 3.4).
  split_bars.png  -- random vs homology grouped bars per comparator (sec 3.5).

Run: uv run scripts/build_result_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures"

M = json.loads((DATA / "metrics_homology.json").read_text())
RAND = json.loads((DATA / "metrics.json").read_text())
RANDC = json.loads((DATA / "metrics_random_comparators.json").read_text())
ENFH = json.loads((DATA / "metrics_enformer_homology.json").read_text())

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENC_DISP = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
FLOOR = 0.224

C_COMP = "#9e9e9e"   # composition (grey)
C_DNA = "#3a7d44"    # DNA encoders (green)
C_ESM = "#c0504d"    # protein/supervised comparator (red)


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
    base = {"kmer_baseline_4": "kmer", "kmer_baseline_6": "kmer6", "codon_baseline": "codon",
            "gc_baseline": "gc", "aa_baseline_1": "aa1", "aa_baseline_2": "aa2", "aa_baseline_3": "aa3"}
    for r in metrics:
        if r.get("task") is not None:
            continue
        ds = str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "")
        if base.get(r.get("model", "")) == src or ds == src or r.get("feature_source") == src:
            return r["test_r2_macro"]
    return None


def f1_of(m, src):
    return _best_cls(m, src) if src in ENCODERS else _cell_cls(m, src)


def r2_of(m, src):
    return _best_reg_enc(m, src) if src in ENCODERS else _cell_reg(m, src)


def _rand_f1(src):
    v = _cell_cls(RANDC, src)
    return v if v is not None else f1_of(RAND, src)


def _no_title(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


# representative comparator cells (composition | DNA | ESM comparator)
CELLS = [("CDS 4-mer", "kmer", C_COMP), ("Codon", "codon", C_COMP), ("AA comp.", "aa_best", C_COMP),
         ("DNABERT-2", "dnabert2", C_DNA), ("NT-v2", "nt_v2", C_DNA),
         ("GENA-LM", "gena_lm", C_DNA), ("HyenaDNA", "hyena_dna", C_DNA),
         ("ESM-2 150M", "esm2_150m", C_ESM), ("ESM-2 650M", "esm2_650m", C_ESM)]


def _aa_best(fn):
    return max(fn(M, s) for s in ("aa1", "aa2", "aa3"))


def _comparator_panel(value_fn, ylabel, ylim, floor, fname):
    labels = [c[0] for c in CELLS]
    colors = [c[2] for c in CELLS]
    vals = [_aa_best(value_fn) if c[1] == "aa_best" else value_fn(M, c[1]) for c in CELLS]
    x = np.arange(len(CELLS))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, vals, color=colors, edgecolor="white",
           hatch=["//" if c[2] == C_ESM else "" for c in CELLS])
    if floor is not None:
        ax.axhline(floor, color="#555", ls="--", lw=0.9)
        ax.text(0.1, floor + 0.008, f"chance {floor:.3f}", fontsize=7.5, color="#555")
    else:
        ax.axhline(0, color="#555", lw=0.8)
    _no_title(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(handles=[Patch(facecolor=C_COMP, label="composition"),
                       Patch(facecolor=C_DNA, label="DNA encoder"),
                       Patch(facecolor=C_ESM, hatch="//", label="ESM-2 (comparator)")],
              fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(fname, [round(v, 3) for v in vals])


def fig_comparator_f1():
    _comparator_panel(f1_of, "5-way family macro-F1", (0, 1.0), FLOOR, "comparator_f1.png")


def fig_comparator_r2():
    _comparator_panel(r2_of, "Ridge-to-GenePT $R^2$", (0, 0.20), None, "comparator_r2.png")


def fig_substrate_collapse():
    cats = ["CDS 4-mer"] + [ENC_DISP[e] for e in ENCODERS] + ["Enformer"]
    kmer_cds = next(r["test_kappa"] for r in M if r.get("task") == "family5" and r["feature_source"] == "kmer")
    kmer_tss = next(r["test_kappa"] for r in M if r.get("task") == "family5" and r["feature_source"] == "enformer_tss_4mer")
    enf_tss = max(r["test_kappa"] for r in ENFH if r.get("task") == "family5")
    cds = [kmer_cds] + [_best_cls(M, e, False, "test_kappa") for e in ENCODERS] + [np.nan]
    tss = [kmer_tss] + [_best_cls(M, e, True, "test_kappa") for e in ENCODERS] + [enf_tss]
    cols = [C_COMP] + [C_DNA] * 4 + [C_ESM]
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.bar(x - w / 2, cds, w, color=cols, edgecolor="white")
    ax.bar(x + w / 2, tss, w, color=cols, alpha=0.45, hatch="//", edgecolor="white")
    _no_title(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("5-way family $\\kappa$ (best pool)")
    ax.set_ylim(0, 0.8)
    ax.legend(handles=[Patch(facecolor="#777", label="CDS"),
                       Patch(facecolor="#777", alpha=0.45, hatch="//", label="TSS window"),
                       Patch(facecolor=C_ESM, label="supervised comparator (TSS)")],
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "substrate_collapse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("substrate_collapse.png CDS:", [round(v, 3) for v in cds[:-1]], "TSS:", [round(v, 3) for v in tss])


def fig_split_bars():
    cells = [("CDS 4-mer", "kmer", C_COMP), ("AA 2-mer", "aa2", C_COMP), ("NT-v2", "nt_v2", C_DNA),
             ("DNABERT-2", "dnabert2", C_DNA), ("ESM-2 650M", "esm2_650m", C_ESM)]
    rand = [_rand_f1(s) for _, s, _ in cells]
    hom = [f1_of(M, s) for _, s, _ in cells]
    cols = [c for *_, c in cells]
    x = np.arange(len(cells))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - w / 2, rand, w, color=cols, alpha=0.5, edgecolor="white")
    ax.bar(x + w / 2, hom, w, color=cols, edgecolor="white")
    ax.axhline(FLOOR, color="#555", ls="--", lw=0.9)
    ax.text(0.0, FLOOR + 0.008, f"chance {FLOOR:.3f}", fontsize=7.5, color="#555")
    _no_title(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.set_ylabel("5-way family macro-F1")
    ax.set_ylim(0, 1.0)
    ax.legend(handles=[Patch(facecolor="#777", alpha=0.5, label="random split"),
                       Patch(facecolor="#777", label="homology split")],
              fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "split_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("split_bars.png  random:", [round(v, 3) for v in rand], "homology:", [round(v, 3) for v in hom])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_comparator_f1()
    fig_comparator_r2()
    fig_substrate_collapse()
    fig_split_bars()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
