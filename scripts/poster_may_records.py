"""The May-protocol accessors the URF poster's figures were built from (frozen).

The poster (Oct 2026) prints the accepted paper's numbers: the May records in
data/metrics_*.json, with GENA-LM's CDS cells swapped for the Sept 30 rerun.
build_result_figures.py now reads the camera-ready records (data/v2), so these
accessors moved here unchanged, and only scripts/build_poster_figures.py uses
them. Do not use them for the paper.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


from data_loader.model_registry import encoder_pools
from linear_trainer.selection import encoder_cells, select_by_val, select_pool

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

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
C_ESM = "#2b5c8a"    # reference models: ESM-2, Enformer (blue, hatched)


# ---------- accessors ----------
def best(recs: dict[str, dict], prefix: str, pools=None) -> str:
    """Validation-selected source among ``prefix + pool`` for the named pools.

    The candidates are named explicitly, not matched by prefix, so E5 cells
    (``tssanchored``, ``centermean``) never join the pick (ledger G14). By
    default the pools are the encoder's own CDS or TSS grid, read from the
    prefix (``"hyena_dna_"``, ``"tss_nt_v2_"``).
    """
    if pools is None:
        tss = prefix.startswith("tss_")
        pools = encoder_pools(prefix[4 if tss else 0:-1], "TSS" if tss else "CDS")
    return select_pool(recs, [prefix + p for p in pools])


# Each encoder's validation-selected family5 pool in the May records (moved
# from headline_cells.py, Oct 1). Computed before build_poster_figures swaps
# GENA-LM's CDS cells into M; it re-picks gena_lm itself.
CLS_RECS = {r["feature_source"]: r for r in M
            if r.get("task") == "family5" and not r.get("shuffled_labels")}
CLS_BEST = {e: best(CLS_RECS, e + "_") for e in ENCODERS}
CLS_BEST_TSS = {e: best(CLS_RECS, f"tss_{e}_") for e in ENCODERS}


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
        if core in encoder_cells(src):
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
             and str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "")
             in encoder_cells(enc)]
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
        if core in encoder_cells(enc):
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



# Pools that read a trained boundary token; undefined for encoders pretrained without one.
BOUNDARY_POOLS = ("specialmean", "clsmean")
NO_BOUNDARY_TOKEN = ("hyena_dna",)
