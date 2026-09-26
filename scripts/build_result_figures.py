#!/usr/bin/env python3
"""Reframed Results figures from the metrics JSONs (no embedded titles --
the LaTeX caption is the title, per repo convention).

  comparator_f1.png / comparator_r2.png  -- composition, DNA encoders, ESM-2
        (comparator) on macro-F1 (sec 3.1) and GenePT R^2 (sec 3.2).
  pooling_heatmap_family5_column.png  -- encoder x pooling macro-F1 heatmap
        (sec 3.3), adaptive label colours for legibility.
  tss_context.png  -- TSS arm in one panel: per model, CDS vs TSS whole-window vs
        TSS-Anchored macro-F1, Enformer pooled to match, plus the anchored chunk's
        best composition baseline (sec 3.4).
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

from data_loader.pool_names import POOL_DISPLAY
from headline_cells import CLS_BEST
from linear_trainer.selection import select_by_val

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures"

M = json.loads((DATA / "metrics_homology.json").read_text())
RAND = json.loads((DATA / "metrics.json").read_text())
RANDC = json.loads((DATA / "metrics_random_comparators.json").read_text())
ENFH = json.loads((DATA / "metrics_enformer_homology.json").read_text())
ANCH = json.loads((DATA / "metrics_tss_anchored.json").read_text())
COMP = json.loads((DATA / "metrics_tss_composition.json").read_text())

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENC_DISP = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
POOLS = ["meanmean", "specialmean", "meanD", "meanG", "maxmean", "clsmean"]
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
    return select_by_val(cells)[metric] if cells else None


def _cls_rec(metrics, src):
    c = [r for r in metrics if r.get("task") == "family5" and not r.get("shuffled_labels")
         and r.get("feature_source") == src]
    return c[0] if c else None


def _cell_cls(metrics, src):
    r = _cls_rec(metrics, src)
    return r["test_macro_f1"] if r else None


def _best_reg_enc(metrics, enc):
    cells = [r for r in metrics if r.get("task") is None and r.get("model") == "linear_probe"
             and not str(r.get("dataset", "")).startswith("dataset_tss_")
             and (str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "") == enc
                  or str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "").startswith(enc + "_"))]
    return select_by_val(cells)["test_r2_macro"] if cells else None


def _best_reg_enc_ctx(metrics, enc, tss=False):
    """Best-pool Ridge R^2 for an encoder within a sequence context (CDS or TSS)."""
    cells = []
    for r in metrics:
        if r.get("task") is not None or r.get("model") != "linear_probe":
            continue
        ds = str(r.get("dataset", ""))
        if ds.startswith("dataset_tss_") != tss:
            continue
        core = ds.replace("dataset_tss_", "").replace("dataset_", "").replace(".parquet", "")
        if core == enc or core.startswith(enc + "_"):
            cells.append(r)
    return select_by_val(cells)["test_r2_macro"] if cells else None


def _reg_rec(metrics, src):
    base = {"kmer_baseline_4": "kmer", "kmer_baseline_6": "kmer6", "codon_baseline": "codon",
            "gc_baseline": "gc", "aa_baseline_1": "aa1", "aa_baseline_2": "aa2", "aa_baseline_3": "aa3"}
    for r in metrics:
        if r.get("task") is not None:
            continue
        ds = str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "")
        if base.get(r.get("model", "")) == src or ds == src or r.get("feature_source") == src:
            return r
    return None


def _cell_reg(metrics, src):
    r = _reg_rec(metrics, src)
    return r["test_r2_macro"] if r else None


def f1_of(m, src):
    return _best_cls(m, src) if src in ENCODERS else _cell_cls(m, src)


def r2_of(m, src):
    return _best_reg_enc(m, src) if src in ENCODERS else _cell_reg(m, src)


def _rand_f1(src):
    v = _cell_cls(RANDC, src)
    return v if v is not None else f1_of(RAND, src)


def _rand_r2(src):
    v = _cell_reg(RANDC, src)
    return v if v is not None else r2_of(RAND, src)


def _no_title(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


# representative comparator cells (composition | DNA | ESM comparator)
CELLS = [("CDS 4-mer", "kmer", C_COMP), ("Codon", "codon", C_COMP), ("AA comp.", "aa_best", C_COMP),
         ("DNABERT-2", "dnabert2", C_DNA), ("NT-v2", "nt_v2", C_DNA),
         ("GENA-LM", "gena_lm", C_DNA), ("HyenaDNA", "hyena_dna", C_DNA),
         ("ESM-2 650M", "esm2_650m", C_ESM)]


def _aa_best(rec_fn, metric):
    """AA-composition k chosen on validation, reported on test."""
    return select_by_val(rec_fn(M, s) for s in ("aa1", "aa2", "aa3"))[metric]


def _label_bars(ax, xs, vals, fmt="{:.2f}", fontsize=7, dy=0.01, rot=0):
    top = ax.get_ylim()[1]
    for xp, v in zip(xs, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if v < 0:  # label below a negative bar, not inside it
            ax.text(xp, v - dy, fmt.format(v), ha="center", va="top",
                    fontsize=fontsize, color="#222", rotation=rot)
            continue
        ax.text(xp, min(v + dy, top - dy), fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize, color="#222", rotation=rot)


def _comparator_panel(value_fn, aa_value, ylabel, ylim, floor, fname, fmt="{:.2f}"):
    labels = [c[0] for c in CELLS]
    colors = [c[2] for c in CELLS]
    vals = [aa_value if c[1] == "aa_best" else value_fn(M, c[1]) for c in CELLS]
    x = np.arange(len(CELLS))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, vals, color=colors, edgecolor="white",
           hatch=["//" if c[2] == C_ESM else "" for c in CELLS])
    ax.set_ylim(*ylim)
    _label_bars(ax, x, vals, fmt=fmt, fontsize=7, dy=ylim[1] * 0.012)
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
                       Patch(facecolor=C_ESM, hatch="//", label="ESM-2 650M (upper bound)")],
              fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(fname, [round(v, 3) for v in vals])


def fig_comparator_f1():
    _comparator_panel(f1_of, _aa_best(_cls_rec, "test_macro_f1"), "5-way family macro-F1", (0, 1.0), FLOOR, "comparator_f1.png")


def fig_comparator_r2():
    _comparator_panel(r2_of, _aa_best(_reg_rec, "test_r2_macro"), "Ridge-to-GenePT $R^2$", (0, 0.20), None, "comparator_r2.png", fmt="{:.3f}")


def fig_tss_context():
    """TSS arm in one panel (homology split): per model, CDS | TSS whole-window | TSS-Anchored.

    Every TSS bar pools the same way across models: encoders at their best whole-window
    rule vs the chunk nearest the TSS; Enformer at its whole-window mean (``trunk_global``)
    vs its central 2,048 bp readout (``trunk_center``). The 4-mer has no anchored bar
    because its chunk depends on the encoder. Diamonds: the validation-selected composition baseline
    (4-mer+GC or 6-mer) of each encoder's anchored chunk.
    """
    enf = {r["feature_source"]: r["test_macro_f1"] for r in ENFH if r.get("task") == "family5"}
    cats = ["4-mer"] + [ENC_DISP[e] for e in ENCODERS] + ["Enformer"]
    hues = [C_COMP] + [C_DNA] * 4 + [C_ESM]
    cds = [_cell_cls(M, "kmer")] + [_best_cls(M, e, False) for e in ENCODERS] + [np.nan]
    whole = [_cell_cls(M, "enformer_tss_4mer")] + [_best_cls(M, e, True) for e in ENCODERS] \
        + [enf["enformer_trunk_global"]]
    anchored = [np.nan] + [_cell_cls(ANCH, f"tss_{e}_tssanchored") for e in ENCODERS] \
        + [enf["enformer_trunk_center"]]
    comp = [select_by_val(_cls_rec(COMP, f"tss_{e}_{v}") for v in ("chunk4mergc", "chunk6mer"))
            ["test_macro_f1"] for e in ENCODERS]

    enf_r2 = {Path(r["dataset"]).stem.replace("dataset_", ""): r["test_r2_macro"]
              for r in ENFH if r.get("task") is None and r.get("dataset")}
    cds_r2 = [_cell_reg(M, "kmer")] + [_best_reg_enc_ctx(M, e, False) for e in ENCODERS] + [np.nan]
    whole_r2 = [_cell_reg(M, "enformer_tss_4mer")] + [_best_reg_enc_ctx(M, e, True) for e in ENCODERS] \
        + [enf_r2["enformer_trunk_global"]]
    # No TSS-Anchored regression probe was run for the encoders; only Enformer's centre readout.
    anchored_r2 = [np.nan] * (1 + len(ENCODERS)) + [enf_r2["enformer_trunk_center"]]

    x = np.arange(len(cats))
    w = 0.27
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 3.8))
    panels = [(axL, cds, whole, anchored, "5-way family macro-F1", (0, 0.95), "{:.2f}", 0.008),
              (axR, cds_r2, whole_r2, anchored_r2, "Ridge-to-GenePT $R^2$", (-0.03, 0.10), "{:.3f}", 0.002)]
    for ax, c, wv, an, ylabel, ylim, fmt, dy in panels:
        ax.bar(x - w, c, w, color=hues, edgecolor="white")
        ax.bar(x, wv, w, color=hues, alpha=0.4, hatch="//", edgecolor="white")
        ax.bar(x + w, an, w, color=hues, alpha=0.75, edgecolor="#222", linewidth=0.9)
        ax.set_ylim(*ylim)
        for xs, vals in ((x - w, c), (x, wv), (x + w, an)):
            _label_bars(ax, xs, vals, fmt=fmt, fontsize=7, dy=dy)
        _no_title(ax)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=9)
    axL.scatter(x[1:5] + w, comp, marker="D", s=22, color="#222", zorder=3)
    axL.axhline(FLOOR, color="#555", ls="--", lw=0.9)
    axR.axhline(0, color="#555", lw=0.8)
    fig.legend(handles=[Patch(facecolor="#777", label="CDS"),
                        Patch(facecolor="#777", alpha=0.4, hatch="//", label="TSS window, whole-window pooling"),
                        Patch(facecolor="#777", alpha=0.75, edgecolor="#222", linewidth=0.9,
                              label="TSS-Anchored (Enformer: central 2,048 bp)"),
                        plt.Line2D([], [], marker="D", color="#222", ls="", markersize=5,
                                   label="best composition of the anchored chunk"),
                        plt.Line2D([], [], color="#555", ls="--", lw=0.9, label=f"chance ({FLOOR:.3f})")],
               fontsize=8.5, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    fig.savefig(OUT / "tss_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("tss_context CDS:", [round(v, 3) for v in cds], "whole:", [round(v, 3) for v in whole],
          "anchored:", [round(v, 3) for v in anchored], "comp:", [round(v, 3) for v in comp])
    print("tss_context R2 CDS:", [round(v, 3) for v in cds_r2], "whole:", [round(v, 3) for v in whole_r2],
          "Enformer centre:", round(anchored_r2[-1], 3))


def fig_split_bars():
    """Random vs homology split, side by side: macro-F1 (left), Ridge R^2 (right)."""
    cells = [("CDS 4-mer", "kmer", C_COMP), ("AA 2-mer", "aa2", C_COMP), ("NT-v2", "nt_v2", C_DNA),
             ("DNABERT-2", "dnabert2", C_DNA), ("ESM-2 650M", "esm2_650m", C_ESM)]
    cols = [c for *_, c in cells]
    labels = [c[0] for c in cells]
    x = np.arange(len(cells))
    w = 0.38
    f1_rand = [_rand_f1(s) for _, s, _ in cells]
    f1_hom = [f1_of(M, s) for _, s, _ in cells]
    r2_rand = [_rand_r2(s) for _, s, _ in cells]
    r2_hom = [r2_of(M, s) for _, s, _ in cells]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.6))

    panels = [(axL, f1_rand, f1_hom, "5-way family macro-F1", (0, 1.0), "{:.2f}", 0.01),
              (axR, r2_rand, r2_hom, "Ridge-to-GenePT $R^2$", (0, 0.4), "{:.3f}", 0.004)]
    for ax, rand, hom, ylabel, ylim, fmt, dy in panels:
        ax.bar(x - w / 2, rand, w, color=cols, alpha=0.5, edgecolor="white")
        ax.bar(x + w / 2, hom, w, color=cols, edgecolor="white")
        _no_title(ax)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        _label_bars(ax, x - w / 2, rand, fmt=fmt, fontsize=7, dy=dy)
        _label_bars(ax, x + w / 2, hom, fmt=fmt, fontsize=7, dy=dy)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
    axL.axhline(FLOOR, color="#555", ls="--", lw=0.9)
    axL.text(-0.45, FLOOR + 0.015, f"chance {FLOOR:.3f}", fontsize=7.5, color="#555")
    fig.legend(handles=[Patch(facecolor="#777", alpha=0.5, label="random split"),
                        Patch(facecolor="#777", label="homology split")],
               fontsize=8.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout()
    fig.savefig(OUT / "split_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("split_bars F1  random:", [round(v, 3) for v in f1_rand], "homology:", [round(v, 3) for v in f1_hom])
    print("split_bars R2  random:", [round(v, 3) for v in r2_rand], "homology:", [round(v, 3) for v in r2_hom])


# Pools that read a trained boundary token; undefined for encoders pretrained without one.
BOUNDARY_POOLS = ("specialmean", "clsmean")
NO_BOUNDARY_TOKEN = ("hyena_dna",)
HEATMAP_LABELS = {"meanmean": "Mean", "specialmean": "Mean\n(Boundary-incl.)", "meanD": "Ends +\nMean",
                  "meanG": "Ends + Mean\n+ Max", "maxmean": "Max", "clsmean": "Mean-CLS"}


def fig_pooling_heatmap():
    """Encoder x pooling macro-F1 heatmap (sec 3.3), homology split.

    Colour is centred on the CDS 4-mer floor (white): green above, red below, darker further away, so the
    claim that pooling alone can move an encoder across composition is visible
    directly. Boxes mark each encoder's validation-selected rule; boundary-token
    rules are n/a for encoders pretrained without such a token.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    floor = _cell_cls(M, "kmer")
    vals = np.full((len(ENCODERS), len(POOLS)), np.nan)
    for i, enc in enumerate(ENCODERS):
        for j, pool in enumerate(POOLS):
            if enc in NO_BOUNDARY_TOKEN and pool in BOUNDARY_POOLS:
                continue
            v = _cell_cls(M, f"{enc}_{pool}")
            if v is not None:
                vals[i, j] = v
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm = TwoSlopeNorm(vmin=min(vmin, floor - 0.01), vcenter=floor, vmax=max(vmax, floor + 0.01))
    # Two hues only: lightness carries the distance from the floor (white = 4-mer).
    cmap = LinearSegmentedColormap.from_list("floor_rg", ["#b2182b", "#ffffff", "#1b7837"])
    cmap.set_bad("#e6e6e6")

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(POOLS)))
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_xticklabels([HEATMAP_LABELS[p] for p in POOLS], fontsize=8.5)
    ax.set_yticklabels([ENC_DISP[e] for e in ENCODERS], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # separate the mean-based rules from Max / Mean-CLS
    ax.axvline(POOLS.index("maxmean") - 0.5, color="white", lw=3)
    for i in range(len(ENCODERS)):
        for j in range(len(POOLS)):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#666")
                continue
            r, g, b, _ = cmap(norm(v))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else "black")
    for i, enc in enumerate(ENCODERS):
        j = POOLS.index(CLS_BEST[enc].rsplit("_", 1)[1])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("macro-F1", fontsize=9)
    ticks = [t for t in (0.3, 0.4, 0.5) if norm.vmin <= t] + [floor] + [t for t in (0.7,) if t <= norm.vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"4-mer {t:.3f}" if t == floor else f"{t:.1f}" for t in ticks], fontsize=8)
    cbar.ax.axhline(floor, color="black", lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "pooling_heatmap_family5_column.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("pooling_heatmap_family5_column.png  selected:", {e: CLS_BEST[e] for e in ENCODERS})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_comparator_f1()
    fig_comparator_r2()
    fig_pooling_heatmap()
    fig_tss_context()
    fig_split_bars()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
