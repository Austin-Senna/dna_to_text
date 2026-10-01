#!/usr/bin/env python3
"""Results figures from the camera-ready records (data/v2; no embedded titles --
the LaTeX caption is the title, per repo convention).

Every cell is read through ``linear_trainer.records`` (one commit, the policy
purge, G7/G2) and picked on validation only. The shuffled-label reference is a
null band from ``data/v2/statistics.json`` (200 label shuffles of one named probe,
G13), not a single shuffled run.
A missing cell raises.

  comparator_f1.png / comparator_r2.png  -- composition, DNA encoders, ESM-2
        (comparator) on macro-F1 (sec 3.1) and GenePT R^2 (sec 3.2).
  comparator_f1_bands.png  -- the macro-F1 panel with the ESM-2 650M null band
        drawn beside the 4-mer's (a candidate: the band depends on the model).
  pooling_heatmap_family5_column.png  -- encoder x pooling macro-F1 heatmap
        (sec 3.3), colour centred on the CDS 4-mer floor (red below, green above),
        boxes on each encoder's validation-selected rule, n/a for boundary-token
        rules on HyenaDNA.
  tss_context.png  -- TSS arm in two panels, macro-F1 (left) and Ridge-to-GenePT
        R^2 (right), all on the disjoint split so the CDS and TSS bars score the
        same test genes: per model, CDS vs TSS whole-window vs TSS-Anchored,
        Enformer pooled to match, plus the anchored chunk's best composition
        baseline (sec 3.4).
  split_bars.png  -- random vs homology grouped bars per comparator, each split
        re-selected on its own validation set (sec 3.5).

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

from linear_trainer import records as R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dna_to_text_paper" / "paper" / "figures"

HOM = DIS = RND = None   # CDS primary; TSS primary (CDS cells too, for pairing); leakage demo
STATS: dict = {}


def load_records() -> None:
    """Read the records and statistics the figures use (set here, not on import)."""
    global HOM, DIS, RND, STATS
    HOM = R.load("splits.json")
    DIS = R.load("splits_tss_disjoint.json")
    RND = R.load("splits_random.json")
    STATS = json.loads((R.V2 / "statistics.json").read_text())
    R.check_inputs(STATS)
    if STATS["stamp"] != R.stamp_of(HOM, DIS, RND):        # G7
        raise R.MixedRecords("statistics.json was built from other records than the figures read")

ENCODERS = list(R.ENCODERS)
ENC_DISP = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
POOLS = ["meanmean", "specialmean", "meanD", "meanG", "maxmean", "clsmean"]
METRIC = {"family5": "test_macro_f1", "genept": "test_r2_macro"}

C_COMP = "#9e9e9e"   # composition (grey)
C_DNA = "#3a7d44"    # DNA encoders (green)
C_ESM = "#2b5c8a"    # reference models: ESM-2, Enformer (blue, hatched)


# ---------- accessors (validation picks only) ----------
def pick(recs, arm, task, name):
    """Source for a named comparator: an encoder (its best pool), the nucleotide or
    amino-acid k-mer (k chosen on validation), or a literal source id."""
    by = R.cells(recs, arm, task)
    if name in ENCODERS:
        return R.best_pool(by, name, arm)
    if name == "nt_kmer":
        return R.best_nt_kmer(by)
    if name == "aa_best":
        return R.best_aa(by)
    return name


def value(recs, arm, task, name):
    return R.cells(recs, arm, task)[pick(recs, arm, task, name)][METRIC[task]]


def null_band(split, task, source):
    return STATS["null_bands"][f"{split}/{task}/{source}"]["band95"]


# The band is the shuffled-label range of one probe (the 4-mer unless named), not
# chance for every model: ESM-2 650M's band sits higher (decided Oct 1: name it).
def _shade_null(ax, band, label=True, name="4-mer", color="#555"):
    ax.axhspan(*band, color=color, alpha=0.15, lw=0)
    if label:
        ax.text(0.1, band[1] + 0.008, f"{name} shuffled-label band {band[0]:.2f}-{band[1]:.2f}",
                fontsize=7.5, color=color)


def _no_title(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


# representative comparator cells (composition | DNA | ESM comparator)
CELLS = [("nt_kmer", C_COMP), ("codon", C_COMP), ("aa_best", C_COMP),
         ("dnabert2", C_DNA), ("nt_v2", C_DNA), ("gena_lm", C_DNA), ("hyena_dna", C_DNA),
         ("esm2_650m", C_ESM)]


def _cell_label(name, src):
    if name == "nt_kmer":
        return "CDS 6-mer" if src == "kmer6" else "CDS 4-mer"
    if name == "aa_best":
        return f"AA {src[-1]}-mer"
    return {"codon": "Codon", "esm2_650m": "ESM-2 650M"}.get(name, ENC_DISP.get(name, name))


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


def _fit_top(ylim, vals):
    """Keep the nominal limits unless a bar would be clipped."""
    return ylim[0], max(ylim[1], 1.12 * max(v for v in vals if np.isfinite(v)))


def _comparator_panel(task, ylabel, ylim, fname, fmt="{:.2f}", esm_band=False):
    srcs = [pick(HOM, "cds", task, n) for n, _ in CELLS]
    labels = [_cell_label(n, src) for (n, _), src in zip(CELLS, srcs)]
    colors = [c for _, c in CELLS]
    vals = [value(HOM, "cds", task, n) for n, _ in CELLS]
    ylim = _fit_top(ylim, vals)
    x = np.arange(len(CELLS))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, vals, color=colors, edgecolor="white",
           hatch=["//" if c == C_ESM else "" for c in colors])
    ax.set_ylim(*ylim)
    _label_bars(ax, x, vals, fmt=fmt, fontsize=7, dy=ylim[1] * 0.012)
    bands = []
    if task == "family5":
        kmer = null_band("splits.json", task, "kmer")
        _shade_null(ax, kmer, label=not esm_band)
        if esm_band:   # both bands in the legend; ESM-2's as a range beside its bar
            esm = null_band("splits.json", task, "esm2_650m")
            ax.errorbar([x[-1] + 0.5], [(esm[0] + esm[1]) / 2], yerr=[[(esm[1] - esm[0]) / 2]],
                        fmt="none", ecolor=C_ESM, elinewidth=2, capsize=4, zorder=4)
            bands = [Patch(facecolor="#555", alpha=0.15,
                           label=f"4-mer shuffled-label band ({kmer[0]:.2f}-{kmer[1]:.2f})"),
                     plt.Line2D([], [], color=C_ESM, lw=2, marker="_", markersize=8,
                                label=f"ESM-2 650M shuffled-label band ({esm[0]:.2f}-{esm[1]:.2f})")]
    else:
        ax.axhline(0, color="#555", lw=0.8)
    _no_title(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(handles=[Patch(facecolor=C_COMP, label="composition"),
                       Patch(facecolor=C_DNA, label="DNA encoder"),
                       Patch(facecolor=C_ESM, hatch="//", label="ESM-2 650M (upper bound)"), *bands],
              fontsize=8, frameon=False,
              # with the bands the legend is too tall for the space above the bars
              **({"loc": "upper center", "bbox_to_anchor": (0.5, -0.22), "ncol": 2} if bands
                 else {"loc": "upper left"}))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(fname, dict(zip(labels, [round(v, 3) for v in vals])))


def fig_comparator_f1():
    _comparator_panel("family5", "5-way family macro-F1", (0, 1.0), "comparator_f1.png")
    _comparator_panel("family5", "5-way family macro-F1", (0, 1.0), "comparator_f1_bands.png", esm_band=True)


def fig_comparator_r2():
    _comparator_panel("genept", "Ridge-to-GenePT $R^2$", (0, 0.20), "comparator_r2.png", fmt="{:.3f}")


def fig_tss_context():
    """TSS arm (disjoint split, both panels): per model, CDS | TSS whole-window | TSS-Anchored.

    Every TSS bar pools the same way across models: encoders at their best whole-window
    rule vs the chunk nearest the TSS; Enformer at its whole-window mean (``trunk_global``)
    vs its central 2,048 bp readout (``trunk_center``). The 4-mer has no anchored bar
    because its chunk depends on the encoder. Diamonds: the validation-selected composition baseline
    (4-mer+GC or 6-mer) of each encoder's anchored chunk.
    """
    cats = ["4-mer"] + [ENC_DISP[e] for e in ENCODERS] + ["Enformer"]
    hues = [C_COMP] + [C_DNA] * 4 + [C_ESM]

    def column(task):
        cds_by, tss_by = R.cells(DIS, "cds", task), R.cells(DIS, "tss", task)
        m = METRIC[task]
        cds = [cds_by["kmer"][m]] + [cds_by[R.best_pool(cds_by, e, "cds")][m] for e in ENCODERS] + [np.nan]
        whole = [tss_by["enformer_tss_4mer"][m]] + [tss_by[R.best_pool(tss_by, e, "tss")][m] for e in ENCODERS] \
            + [tss_by["enformer_trunk_global"][m]]
        anchored = [np.nan] + [tss_by[f"tss_{e}_tssanchored"][m] for e in ENCODERS] \
            + [tss_by["enformer_trunk_center"][m]]
        comp = [tss_by[R.pick(tss_by, [f"tss_{e}_chunk4mergc", f"tss_{e}_chunk6mer"])][m]
                for e in ENCODERS]
        return cds, whole, anchored, comp

    cds, whole, anchored, comp = column("family5")
    cds_r2, whole_r2, anchored_r2, comp_r2 = column("genept")
    band = null_band("splits_tss_disjoint.json", "family5", "enformer_tss_4mer")

    x = np.arange(len(cats))
    w = 0.27
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 3.8))
    # R^2 limits follow the data: TSS cells sit below zero, by an amount the records set.
    r2_vals = [v for v in cds_r2 + whole_r2 + anchored_r2 + comp_r2 if np.isfinite(v)]
    lo, hi = min(min(r2_vals), 0.0), max(max(r2_vals), 0.0)
    pad = 0.2 * (hi - lo)
    r2_lim = (lo - pad, hi + pad)
    panels = [(axL, cds, whole, anchored, "5-way family macro-F1", (0, 0.95), "{:.2f}", 0.008),
              (axR, cds_r2, whole_r2, anchored_r2, "Ridge-to-GenePT $R^2$", r2_lim, "{:.3f}",
               0.015 * (r2_lim[1] - r2_lim[0]))]
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
    axR.scatter(x[1:5] + w, comp_r2, marker="D", s=22, color="#222", zorder=3)
    _shade_null(axL, band, label=False)
    axR.axhline(0, color="#555", lw=0.8)
    fig.legend(handles=[Patch(facecolor="#777", label="CDS"),
                        Patch(facecolor="#777", alpha=0.4, hatch="//", label="TSS window, whole-window pooling"),
                        Patch(facecolor="#777", alpha=0.75, edgecolor="#222", linewidth=0.9,
                              label="TSS-Anchored (Enformer: central 2,048 bp)"),
                        plt.Line2D([], [], marker="D", color="#222", ls="", markersize=5,
                                   label="best composition of the anchored chunk"),
                        Patch(facecolor="#555", alpha=0.15, label=f"TSS 4-mer shuffled-label band ({band[0]:.2f}-{band[1]:.2f})")],
               fontsize=8.5, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    fig.savefig(OUT / "tss_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("tss_context CDS:", [round(v, 3) for v in cds], "whole:", [round(v, 3) for v in whole],
          "anchored:", [round(v, 3) for v in anchored], "comp:", [round(v, 3) for v in comp])
    print("tss_context R2 CDS:", [round(v, 3) for v in cds_r2], "whole:", [round(v, 3) for v in whole_r2],
          "anchored:", [round(v, 3) for v in anchored_r2], "comp:", [round(v, 3) for v in comp_r2])


def fig_split_bars():
    """Random vs homology split, side by side: macro-F1 (left), Ridge R^2 (right)."""
    cells = [("nt_kmer", C_COMP), ("aa_best", C_COMP), ("nt_v2", C_DNA), ("dnabert2", C_DNA),
             ("esm2_650m", C_ESM)]
    cols = [c for _, c in cells]
    # Each split and task picks its own k: name it only where all four picks agree.
    generic = {"nt_kmer": "CDS k-mer", "aa_best": "AA k-mer"}
    labels = []
    for n, _ in cells:
        picks = {pick(r, "cds", t, n) for r in (HOM, RND) for t in ("family5", "genept")}
        labels.append(generic[n] if n in generic and len(picks) > 1 else _cell_label(n, picks.pop()))
    x = np.arange(len(cells))
    w = 0.38
    # Each split re-selects its pools and k on its own validation set.
    f1_rand = [value(RND, "cds", "family5", n) for n, _ in cells]
    f1_hom = [value(HOM, "cds", "family5", n) for n, _ in cells]
    r2_rand = [value(RND, "cds", "genept", n) for n, _ in cells]
    r2_hom = [value(HOM, "cds", "genept", n) for n, _ in cells]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.6))

    panels = [(axL, f1_rand, f1_hom, "5-way family macro-F1", (0, 1.0), "{:.2f}", 0.01),
              (axR, r2_rand, r2_hom, "Ridge-to-GenePT $R^2$", _fit_top((0, 0.4), r2_rand + r2_hom),
               "{:.3f}", 0.004)]
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
    _shade_null(axL, null_band("splits.json", "family5", "kmer"), name="homology-split 4-mer")
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

    by = R.cells(HOM, "cds", "family5")
    floor = by["kmer"]["test_macro_f1"]
    vals = np.full((len(ENCODERS), len(POOLS)), np.nan)
    for i, enc in enumerate(ENCODERS):
        for j, pool in enumerate(POOLS):
            if enc in NO_BOUNDARY_TOKEN and pool in BOUNDARY_POOLS:
                continue
            vals[i, j] = by[f"{enc}_{pool}"]["test_macro_f1"]
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
        j = POOLS.index(R.best_pool(by, enc, "cds").rsplit("_", 1)[1])
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
    print("pooling_heatmap_family5_column.png  selected:", {e: R.best_pool(by, e, "cds") for e in ENCODERS})


def main():
    load_records()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_comparator_f1()
    fig_comparator_r2()
    fig_pooling_heatmap()
    fig_tss_context()
    fig_split_bars()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
