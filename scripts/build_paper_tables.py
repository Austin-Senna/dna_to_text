#!/usr/bin/env python3
"""Generate LaTeX table-body fragments for the dna_to_text_paper manuscript
from the homology-split metrics.

The manuscript keeps its own ``\\begin{table}`` / ``\\processtable{caption}`` /
``\\begin{tabular*}`` / header / source-note wrappers; this script only writes
the data rows (and ``\\multicolumn`` section separators) that go between the
header ``\\midrule`` and the closing ``\\botrule``. The paper ``\\input``s each
fragment. Re-run to refresh every number; nothing is hand-transcribed.

    uv run scripts/build_paper_tables.py

Primary split: homology-aware MMseqs2, 40% identity (``data/metrics_homology.json``).
Alpha for ridge is selected by validation macro-R^2 (``select_by == "r2"``).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dna_to_text_paper" / "paper" / "tables"

POOLS = ["meanmean", "specialmean", "maxmean", "clsmean", "meanD", "meanG"]
ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENC_DISPLAY = {
    "dnabert2": "DNABERT-2",
    "nt_v2": "NT-v2",
    "gena_lm": "GENA-LM",
    "hyena_dna": "HyenaDNA",
}
# composition + protein-LM single-cell sources, in display order
COMPOSITION = [
    ("kmer", "CDS 4-mer"),
    ("kmer6", "CDS 6-mer"),
    ("codon", "Codon"),
    ("aa1", "AA 1-mer"),
    ("aa2", "AA 2-mer"),
    ("aa3", "AA 3-mer"),
]
ESM = [("esm2_650m", "ESM-2 650M")]
# Subset shown in the main best-cell tables; CDS 6-mer stays in the appendix
# full matrices only. (GC+length is dropped from the study entirely.)
MAIN_COMPOSITION = [c for c in COMPOSITION if c[0] != "kmer6"]
# regression baseline model name -> raw source id
BASELINE_MODEL = {
    "kmer_baseline_4": "kmer",
    "kmer_baseline_6": "kmer6",
    "codon_baseline": "codon",
    "gc_baseline": "gc",
    "aa_baseline_1": "aa1",
    "aa_baseline_2": "aa2",
    "aa_baseline_3": "aa3",
}
CDS_4MER = "kmer"
TSS_4MER = "enformer_tss_4mer"  # the 4-mer-on-TSS-window baseline (misleading raw name)


def load(name: str):
    return json.loads((DATA / name).read_text())


# ---------- source parsing ----------
def split_enc_pool(rest: str):
    """rest like 'nt_v2_meanD' -> ('nt_v2','meanD')."""
    for enc in ENCODERS:
        if rest == enc:
            return enc, "base"
        if rest.startswith(enc + "_"):
            return enc, rest[len(enc) + 1 :]
    return None, None


def parse_source(raw: str) -> dict:
    """Map a raw source id to display metadata."""
    if raw == TSS_4MER:
        return dict(cat="tss-baseline", enc=None, pool=None, ctx="TSS", display="TSS 4-mer")
    if raw.startswith("tss_"):
        enc, pool = split_enc_pool(raw[4:])
        return dict(cat="dna-lm", enc=enc, pool=pool, ctx="TSS", display=ENC_DISPLAY[enc])
    for rid, disp in ESM:
        if raw == rid:
            return dict(cat="protein-lm", enc=raw, pool=None, ctx="CDS", display=disp)
    for rid, disp in COMPOSITION:
        if raw == rid:
            cat = "aa-comp" if rid.startswith("aa") else "dna-comp"
            return dict(cat=cat, enc=raw, pool=None, ctx="CDS", display=disp)
    enc, pool = split_enc_pool(raw)
    if enc is not None:
        return dict(cat="dna-lm", enc=enc, pool=pool, ctx="CDS", display=ENC_DISPLAY[enc])
    raise ValueError(f"unknown source id: {raw}")


def reg_raw(rec: dict) -> str:
    """Raw source id for a regression record."""
    model = rec.get("model")
    if model == "linear_probe":
        return rec["dataset"].replace("dataset_", "").replace(".parquet", "")
    if model in BASELINE_MODEL:
        return BASELINE_MODEL[model]
    raise ValueError(f"unknown regression model: {model}")


# ---------- formatting ----------
def f(x, d=4):
    return f"{x:.{d}f}"


def sgn(x, d=4):
    return f"{x:+.{d}f}"


def tt(s):
    return r"\texttt{" + s + "}" if s and s != "base" else "---"


def bold(s):
    return r"\textbf{" + s + "}"


def alpha_str(a):
    a = float(a)
    return str(int(a)) if a >= 1 else ("%g" % a)


# Per-table tabular wrapper specs. Each fragment is a SELF-CONTAINED tabular so
# no alignment (\\, \midrule, \multicolumn, \botrule) ever spans the \input edge
# -- those primitives do cross-boundary lookahead and misfire otherwise. The
# paper keeps the caption + source-note via \processtable{cap}{\input{..}}{note}.
SPECS = {
    "family5_main": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7.5}{8.8}\selectfont",
                         width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}llrrrr@{}",
                         header=r"Source & Pool & F1 & $\Delta$F1 & $\kappa$ & Acc."),
    "ridge_main": dict(setup=r"\setlength{\tabcolsep}{2.5pt}",
                       width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}llrrr@{}",
                       header=r"Source & Pool & $R^2$ & $\Delta$ & Cos."),
    "cds_tss": dict(setup=r"\setlength{\tabcolsep}{2.5pt}\fontsize{8}{9.5}\selectfont",
                    width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrrrr@{}",
                    header=r"Source & F1 & $\Delta$F1 & $\kappa$ & $R^2$ & $\Delta R^2$"),
    "protein_comparison": dict(setup=r"\setlength{\tabcolsep}{3pt}",
                               width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrr@{}",
                               header=r"Source & Macro-F1 & $\kappa$ & GenePT $R^2$"),
    "leakage": dict(setup=r"\setlength{\tabcolsep}{3pt}",
                    width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrr@{}",
                    header=r"Source & Random F1 & Homology F1 & $\Delta$"),
    "split_comparison": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7.5}{9}\selectfont",
                             width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lrrr|rrr@{}",
                             header=r"Source & F1 (rand) & F1 (hom) & $\Delta$F1 & $R^2$ (rand) & $R^2$ (hom) & $\Delta R^2$"),
    "split_comparison_full": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7}{8.5}\selectfont",
                                  width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lrrrr@{}",
                                  header=r"Source & F1 (rand) & F1 (hom) & $R^2$ (rand) & $R^2$ (hom)"),
    "s_pooling_full": dict(setup=r"\setlength{\tabcolsep}{1pt}\fontsize{5}{6}\selectfont",
                           width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}llrrrr@{}",
                           header=r"Encoder & Pooling & Macro-F1 & $\kappa$ & $\Delta\kappa$ & Accuracy"),
    "s_regression_full": dict(setup=r"\setlength{\tabcolsep}{1pt}\fontsize{5}{6}\selectfont",
                              width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}llrrrr@{}",
                              header=r"Feature source & Pooling & $R^2$ macro & $\Delta$ & Mean cosine & $\alpha$"),
    "s_pooling_full_random": dict(setup=r"\setlength{\tabcolsep}{1pt}\fontsize{5}{6}\selectfont",
                                  width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}llrrr@{}",
                                  header=r"Encoder & Pooling & Macro-F1 & $\Delta$F1 & Accuracy"),
    "s_regression_full_random": dict(setup=r"\setlength{\tabcolsep}{1pt}\fontsize{5}{6}\selectfont",
                                     width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}llrrrr@{}",
                                     header=r"Feature source & Pooling & $R^2$ macro & $\Delta$ & Mean cosine & $\alpha$"),
    "s_seed_sensitivity": dict(setup="", width=r"0.9\columnwidth",
                               cols=r"@{\extracolsep{\fill}}lrr@{}",
                               header=r"Cell & Macro-F1 (4 seeds) & GenePT $R^2$ (4 seeds)"),
    "s_homology70": dict(setup="", width=r"0.9\columnwidth",
                         cols=r"@{\extracolsep{\fill}}lrrr@{}",
                         header=r"Cell & Macro-F1 & $\kappa$ & GenePT $R^2$"),
    "s_cds_tss_paired": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{6.5}{7.5}\selectfont",
                             width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lccr@{}",
                             header=r"Encoder & $\Delta$Macro-F1 [95\% CI] & $\Delta R^2$ [95\% CI] & $P(\textrm{CDS}{>}\textrm{TSS})$"),
    "s_headline_ci_cls": dict(setup=r"\setlength{\tabcolsep}{3pt}\fontsize{7.5}{9}\selectfont",
                              width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}llcc@{}",
                              header=r"Source & Pool & Macro-F1 [95\% CI] & $\kappa$ [95\% CI]"),
    "s_headline_ci_reg": dict(setup=r"\setlength{\tabcolsep}{3pt}\fontsize{7.5}{9}\selectfont",
                              width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}llc@{}",
                              header=r"Source & Pool & GenePT $R^2$ [95\% CI]"),
    "s_tss_disjoint": dict(setup=r"\setlength{\tabcolsep}{3pt}",
                           width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrr@{}",
                           header=r"Source & Homology F1 & Disjoint F1 & $\Delta$"),
    "s_paired_diff": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7}{8.4}\selectfont",
                          width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lcr@{}",
                          header=r"Comparison & $\Delta$ [95\% CI] & $P(A{>}B)$"),
    "s_ridge_robust": dict(setup=r"\setlength{\tabcolsep}{3pt}",
                           width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrrr@{}",
                           header=r"Method & Macro-$R^2$ & Pooled-$R^2$ & Retr.@5 & Med.\ rank"),
}


def write(key: str, body: str):
    OUT.mkdir(parents=True, exist_ok=True)
    s = SPECS[key]
    block = (
        "{" + s["setup"] + r"\begin{tabular*}{" + s["width"] + "}{" + s["cols"] + r"}\toprule" + "\n"
        + s["header"] + r" \\\midrule" + "\n"
        + body.rstrip() + "\n"
        + r"\botrule" + "\n"
        + r"\end{tabular*}}" + "\n"
    )
    (OUT / f"{key}.tex").write_text(block)
    print(f"wrote {key}.tex ({block.count(chr(10))} lines)")


# ---------- load + index ----------
M = load("metrics_homology.json")
M70 = load("metrics_homology70.json")
SEED = load("seed_sensitivity/summary.json")
BOOT = load("bootstrap_metrics.json")
RAND = load("metrics.json")  # random-stratified split: DNA encoders + 4-mer + TSS
RANDC = load("metrics_random_comparators.json")  # composition + ESM-2 on random split
ENFH = load("metrics_enformer_homology.json")  # Enformer re-probed on the homology split

# classification index: feature_source -> record (non-shuffled), plus shuffled
CLS = {}
CLS_SHUF = None
for r in M:
    if r.get("task") != "family5":
        continue
    if r.get("shuffled_labels"):
        CLS_SHUF = r
    else:
        CLS[r["feature_source"]] = r

# regression index: raw source -> record
REG = {}
for r in M:
    if r.get("task") is not None:
        continue
    REG[reg_raw(r)] = r

# random-split indices for the appendix mirror tables. DNA-encoder cells come
# from metrics.json; composition + ESM from metrics_random_comparators.json
# (iterated last so the comparator re-run wins on shared keys, e.g. the 4-mer).
# Note: cached random DNA-LM runs lack test_kappa, so the random classification
# mirror reports macro-F1 (+ delta over the random 4-mer) and accuracy only.
CLS_RAND = {}
for r in list(RAND) + list(RANDC):
    if r.get("task") == "family5" and not r.get("shuffled_labels"):
        CLS_RAND[r["feature_source"]] = r
REG_RAND = {}
for r in list(RAND) + list(RANDC):
    if r.get("task") is not None:
        continue
    if r.get("model") == "linear_probe" and not r.get("dataset"):
        continue  # stray summary row with no dataset
    try:
        REG_RAND[reg_raw(r)] = r
    except (KeyError, ValueError):
        continue  # anti-baselines / rows we don't tabulate here


def cls_best_pool(enc, ctx="CDS"):
    """Best-macro-F1 pool record for an encoder in a context."""
    prefix = ("tss_" if ctx == "TSS" else "") + enc + "_"
    cells = [(fsrc, rec) for fsrc, rec in CLS.items() if fsrc.startswith(prefix)]
    if not cells:
        return None, None
    fsrc, rec = max(cells, key=lambda kv: kv[1]["test_macro_f1"])
    return fsrc.split("_")[-1], rec


def reg_best_pool(enc, ctx="CDS"):
    prefix = ("tss_" if ctx == "TSS" else "") + enc + "_"
    cells = [(s, rec) for s, rec in REG.items() if s.startswith(prefix)]
    if not cells:
        return None, None
    s, rec = max(cells, key=lambda kv: kv[1]["test_r2_macro"])
    return s.split("_")[-1], rec


# ===================================================================
# Table 1: best 5-way family classification cell (main text, 3 dp)
# Columns: Source & Pool & F1 & kappa & Dkappa & Acc.
# ===================================================================
def build_family5_main():
    base_f1 = CLS[CDS_4MER]["test_macro_f1"]
    rows = []  # (display, pool_tex, f1, df1, kappa, acc, is_control)
    rows.append(("Shuffled labels", "---", CLS_SHUF["test_macro_f1"],
                 CLS_SHUF["test_macro_f1"] - base_f1, CLS_SHUF["test_kappa"],
                 CLS_SHUF["test_accuracy"], True))
    for rid, disp in MAIN_COMPOSITION:
        r = CLS[rid]
        rows.append((disp, "---", r["test_macro_f1"], r["test_macro_f1"] - base_f1,
                     r["test_kappa"], r["test_accuracy"], False))
    for enc in ENCODERS:
        pool, r = cls_best_pool(enc)
        rows.append((ENC_DISPLAY[enc], tt(pool), r["test_macro_f1"], r["test_macro_f1"] - base_f1,
                     r["test_kappa"], r["test_accuracy"], False))
    # bold per-column max among non-controls; ESM-2 added below as upper bound
    body = render_main_rows(rows, dp=3, cols=("f1", "df1", "kappa", "acc"),
                            rule_after=1 + len(MAIN_COMPOSITION))
    esm = CLS["esm2_650m"]
    esm_row = (f"ESM-2 650M & --- & {f(esm['test_macro_f1'], 3)} & {sgn(esm['test_macro_f1'] - base_f1, 3)} "
               f"& {f(esm['test_kappa'], 3)} & {f(esm['test_accuracy'], 3)} \\\\")
    return body + "\n" + r"\midrule" + "\n" + esm_row


def render_main_rows(rows, dp, cols, rule_after=None):
    # rows: (display, pool, f1, df1, kappa, acc, is_control)
    # rule_after: insert a \midrule after this many leading (baseline) rows.
    vals = {c: [] for c in cols}
    for (disp, pool, v_f1, v_df1, v_k, v_acc, ctrl) in rows:
        m = dict(f1=v_f1, df1=v_df1, kappa=v_k, acc=v_acc)
        for c in cols:
            vals[c].append((m[c], ctrl))
    best = {c: max((v for v, ctrl in vals[c] if not ctrl)) for c in cols}
    out = []
    for i, (disp, pool, v_f1, v_df1, v_k, v_acc, ctrl) in enumerate(rows):
        m = dict(f1=v_f1, df1=v_df1, kappa=v_k, acc=v_acc)
        cells = []
        for c in ("f1", "df1", "kappa", "acc"):
            txt = sgn(m[c], dp) if c == "df1" else f(m[c], dp)
            if (not ctrl) and m[c] == best[c]:
                txt = bold(txt)
            cells.append(txt)
        out.append(f"{disp} & {pool} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
        if rule_after is not None and i == rule_after - 1:
            out.append(r"\midrule")
    return "\n".join(out)


# ===================================================================
# Table 2: best Ridge-to-GenePT cell (main text, 3 dp)
# Columns: Source & Pool & R^2 & Delta & Cos.
# (no shuffled-Y control in homology data)
# ===================================================================
def build_ridge_main():
    base_r2 = REG[CDS_4MER]["test_r2_macro"]
    rows = []  # (display, pool, r2, delta, cos)
    for rid, disp in MAIN_COMPOSITION:
        r = REG[rid]
        rows.append((disp, "---", r["test_r2_macro"], r["test_r2_macro"] - base_r2,
                     r["test_mean_cosine"]))
    for enc in ENCODERS:
        pool, r = reg_best_pool(enc)
        rows.append((ENC_DISPLAY[enc], tt(pool), r["test_r2_macro"],
                     r["test_r2_macro"] - base_r2, r["test_mean_cosine"]))
    # bold per-column max among non-controls; ESM-2 added below as upper bound
    best_r2 = max(r[2] for r in rows)
    best_cos = max(r[4] for r in rows)
    out = []
    for i, (disp, pool, r2, delta, cos) in enumerate(rows):
        r2t = bold(f(r2, 3)) if r2 == best_r2 else f(r2, 3)
        cost = bold(f(cos, 3)) if cos == best_cos else f(cos, 3)
        out.append(f"{disp} & {pool} & {r2t} & {sgn(delta,3)} & {cost} \\\\")
        if i == len(MAIN_COMPOSITION) - 1:
            out.append(r"\midrule")
    esm = REG["esm2_650m"]
    esm_row = (f"ESM-2 650M & --- & {f(esm['test_r2_macro'], 3)} "
               f"& {sgn(esm['test_r2_macro'] - base_r2, 3)} & {f(esm['test_mean_cosine'], 3)} \\\\")
    return "\n".join(out) + "\n" + r"\midrule" + "\n" + esm_row


# ===================================================================
# Table 3: substrate ablation CDS vs TSS (main text)
# Columns: Source & F1 & DF1 & kappa & R^2 & DR^2.
# Headline metric is macro-F1 (matches the best-pool/C selection criterion);
# kappa is reported as a secondary column. TSS 4-mer R^2 from the
# enformer_tss_4mer regression record; TSS DR^2 stays '---' because the TSS
# 4-mer R^2 is itself below noise (<0), so a within-TSS R^2 delta would be
# anchored to a sub-zero baseline; absolute TSS R^2 (all at noise) is reported
# instead (regression deltas are within CDS only, per Methods).
# Enformer: a single best-F1 variant (trunk_center) is reported across
# F1/kappa/R^2 so the columns never mix variants.
# ===================================================================
def build_cds_tss():
    out = []
    # --- CDS ---
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Coding sequence (CDS)}}\\")
    base_f1 = CLS[CDS_4MER]["test_macro_f1"]
    base_k = CLS[CDS_4MER]["test_kappa"]
    base_r2 = REG[CDS_4MER]["test_r2_macro"]
    out.append(f"\\quad 4-mer & {f(base_f1,3)} & {sgn(0,3)} & {f(base_k,3)} & {f(base_r2,3)} & {sgn(0,3)} \\\\")
    cds_rows = []
    for enc in ENCODERS:
        _, kc = cls_best_pool(enc, "CDS")
        _, rc = reg_best_pool(enc, "CDS")
        f1 = kc["test_macro_f1"]; k = kc["test_kappa"]; r2 = rc["test_r2_macro"]
        cds_rows.append((enc, f1, f1 - base_f1, k, r2, r2 - base_r2))
    bf = max(r[1] for r in cds_rows); br2 = max(r[4] for r in cds_rows)
    for enc, f1, df1, k, r2, dr2 in cds_rows:
        name = ENC_DISPLAY[enc]
        f1t = bold(f(f1, 3)) if f1 == bf else f(f1, 3)
        df1t = bold(sgn(df1, 3)) if f1 == bf else sgn(df1, 3)
        r2t = bold(f(r2, 3)) if r2 == br2 else f(r2, 3)
        dr2t = bold(sgn(dr2, 3)) if r2 == br2 else sgn(dr2, 3)
        if enc == "nt_v2":
            name = bold(name)
        out.append(f"\\quad {name} & {f1t} & {df1t} & {f(k,3)} & {r2t} & {dr2t} \\\\")
    out.append(r"\midrule")
    # --- TSS ---
    out.append(r"\multicolumn{6}{@{}l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    tb_f1 = CLS[TSS_4MER]["test_macro_f1"]
    tb_k = CLS[TSS_4MER]["test_kappa"]
    tb_r2 = REG[TSS_4MER]["test_r2_macro"]
    out.append(f"\\quad 4-mer & {f(tb_f1,3)} & {sgn(0,3)} & {f(tb_k,3)} & {f(tb_r2,3)} & {sgn(0,3)} \\\\")
    tss_rows = []
    for enc in ENCODERS:
        _, kc = cls_best_pool(enc, "TSS")
        _, rc = reg_best_pool(enc, "TSS")
        f1 = kc["test_macro_f1"]; k = kc["test_kappa"]; r2 = rc["test_r2_macro"]
        tss_rows.append((enc, f1, f1 - tb_f1, k, r2, r2 - tb_r2))
    bf = max(r[1] for r in tss_rows)
    for enc, f1, df1, k, r2, dr2 in tss_rows:
        name = ENC_DISPLAY[enc]
        f1t = bold(f(f1, 3)) if f1 == bf else f(f1, 3)
        df1t = bold(sgn(df1, 3)) if f1 == bf else sgn(df1, 3)
        if enc == "nt_v2":
            name = bold(name)
        out.append(f"\\quad {name} & {f1t} & {df1t} & {f(k,3)} & {f(r2,3)} & {sgn(dr2,3)} \\\\")
    enf_cls = max((r for r in ENFH if r.get("task") == "family5"),
                  key=lambda r: r["test_macro_f1"])
    enf_var = enf_cls["feature_source"]  # e.g. enformer_trunk_center
    enf_reg = next(r for r in ENFH if r.get("task") is None and enf_var in (r.get("dataset") or ""))
    enf_f1 = enf_cls["test_macro_f1"]; enf_k = enf_cls["test_kappa"]; enf_r2 = enf_reg["test_r2_macro"]
    out.append(f"\\quad Enformer$^\\dagger$ & {f(enf_f1,3)} & {sgn(enf_f1-tb_f1,3)} & {f(enf_k,3)} & {f(enf_r2,3)} & {sgn(enf_r2-tb_r2,3)} \\\\")
    return "\n".join(out)


# ===================================================================
# Table A1: full classification matrix (appendix, 4 dp), grouped CDS / TSS
# ===================================================================
def build_pooling_full():
    out = []
    base_k = CLS[CDS_4MER]["test_kappa"]
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Coding sequence (CDS)}}\\")

    def row(disp, pool, r):
        return (f"{disp} & {pool} & {f(r['test_macro_f1'])} & {f(r['test_kappa'])} "
                f"& {sgn(r['test_kappa']-base_k)} & {f(r['test_accuracy'])} \\\\")

    out.append(row("Shuffled labels", "---", CLS_SHUF))
    for rid, disp in COMPOSITION:
        out.append(row(disp, "baseline" if rid == "kmer" else "---", CLS[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"{enc}_{pool}"
            if fsrc in CLS:
                out.append(row(ENC_DISPLAY[enc], tt(pool), CLS[fsrc]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Protein language model (translated CDS)}}\\")
    for rid, disp in ESM:
        out.append(row(disp, "---", CLS[rid]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    tbk = CLS[TSS_4MER]["test_kappa"]

    def trow(disp, pool, r):
        return (f"{disp} & {pool} & {f(r['test_macro_f1'])} & {f(r['test_kappa'])} "
                f"& {sgn(r['test_kappa']-tbk)} & {f(r['test_accuracy'])} \\\\")

    out.append(trow("TSS 4-mer", "baseline", CLS[TSS_4MER]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"tss_{enc}_{pool}"
            if fsrc in CLS:
                out.append(trow(ENC_DISPLAY[enc], tt(pool), CLS[fsrc]))
    return "\n".join(out)


# ===================================================================
# Table A2: full regression matrix (appendix, 4 dp), grouped CDS / TSS
# Columns: Feature source & Pooling & R^2 macro & Delta & Mean cosine & alpha.
# TSS has no 4-mer regression baseline -> Delta '---' for TSS rows.
# ===================================================================
def build_regression_full():
    out = []
    base_r2 = REG[CDS_4MER]["test_r2_macro"]
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Coding sequence (CDS)}}\\")

    def row(disp, pool, r, delta=True):
        d = sgn(r["test_r2_macro"] - base_r2) if delta else "---"
        return (f"{disp} & {pool} & {f(r['test_r2_macro'])} & {d} "
                f"& {f(r['test_mean_cosine'])} & {alpha_str(r['alpha'])} \\\\")

    for rid, disp in COMPOSITION:
        out.append(row(disp, "baseline" if rid == "kmer" else "---", REG[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"{enc}_{pool}"
            if s in REG:
                out.append(row(ENC_DISPLAY[enc], tt(pool), REG[s]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Protein language model (translated CDS)}}\\")
    for rid, disp in ESM:
        out.append(row(disp, "---", REG[rid]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    out.append(row("TSS 4-mer", "baseline", REG[TSS_4MER], delta=False))
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"tss_{enc}_{pool}"
            if s in REG:
                out.append(row(ENC_DISPLAY[enc], tt(pool), REG[s], delta=False))
    return "\n".join(out)


# ===================================================================
# Table A3: split-seed sensitivity (appendix)
# seed 42 = primary metrics; seeds 1/7/123 = summary.json. Range over the four.
# Columns: cell & macro-F1 range & R^2 range.
# ===================================================================
def build_seed_sensitivity():
    seeds = ["1", "7", "123"]
    # (display, cls_source, reg_source)
    cells = [
        ("ESM-2 650M", "esm2_650m", "esm2_650m"),
        ("Best DNA-LM", "nt_v2_meanG", "dnabert2_meanD"),
        ("AA-composition", "aa2", "aa3"),
        ("TSS (DNABERT-2)", "tss_dnabert2_meanmean", "tss_dnabert2_meanmean"),
    ]
    out = []
    for disp, cs, rs in cells:
        f1s = [CLS[cs]["test_macro_f1"]] + [SEED[s]["cls"][cs] for s in seeds]
        r2s = [REG[rs]["test_r2_macro"]] + [SEED[s]["reg"][rs] for s in seeds]
        out.append(f"{disp} & {f(min(f1s),3)}--{f(max(f1s),3)} "
                   f"& {f(min(r2s),3)}--{f(max(r2s),3)} \\\\")
    return "\n".join(out)


# ===================================================================
# Table A4: 70%-identity supplementary split (appendix)
# Columns: cell & macro-F1 & kappa & R^2 (R^2 blank for cls-only rows handled).
# Built from the 11 cells re-probed at 70% id.
# ===================================================================
def build_homology70():
    cls70 = {}
    reg70 = {}
    for r in M70:
        if r.get("task"):
            cls70[r["feature_source"]] = r
        else:
            reg70[reg_raw(r)] = r
    # (display, cls_source, reg_source)
    cells = [
        ("ESM-2 650M", "esm2_650m", "esm2_650m"),
        ("Best DNA-LM", "nt_v2_meanG", "dnabert2_meanD"),
        ("AA-composition", "aa2", "aa3"),
        ("CDS 4-mer", "kmer", None),
        ("TSS (DNABERT-2)", "tss_dnabert2_meanmean", "tss_dnabert2_meanmean"),
    ]
    out = []
    for disp, cs, rs in cells:
        c = cls70[cs]
        r2 = f(reg70[rs]["test_r2_macro"], 3) if rs and rs in reg70 else "---"
        out.append(f"{disp} & {f(c['test_macro_f1'],3)} & {f(c['test_kappa'],3)} & {r2} \\\\")
    return "\n".join(out)


# ===================================================================
# Table A5: CDS-vs-TSS paired bootstrap CIs (appendix)
# Columns: Encoder & DF1 [95% CI] & DR^2 [95% CI] & P(CDS>TSS).
# ===================================================================
def build_cds_tss_paired():
    pc = BOOT["paired"]["classification"]
    pr = BOOT["paired"]["regression"]
    rows = [(ENC_DISPLAY[e], f"{e} CDS - TSS") for e in ENCODERS]
    rows.append(("4-mer (control)", "kmer CDS - TSS 4mer"))
    out = []
    for disp, key in rows:
        c = pc[key]; r = pr[key]
        f1 = f"{sgn(c['delta_macro_f1_point'],3)} [{sgn(c['delta_macro_f1_ci95'][0],3)}, {sgn(c['delta_macro_f1_ci95'][1],3)}]"
        r2 = f"{sgn(r['delta_r2_macro_point'],3)} [{sgn(r['delta_r2_macro_ci95'][0],3)}, {sgn(r['delta_r2_macro_ci95'][1],3)}]"
        p = f"{c['frac_A_gt_B_f1']:.3f}"
        out.append(f"{disp} & {f1} & {r2} & {p} \\\\")
    return "\n".join(out)


# ===================================================================
# Tables A6/A7: headline-cell 95% bootstrap CIs (appendix). Makes good the
# Methods promise that every headline cell carries a test-set CI, without
# crowding the main best-cell tables. The POINT estimate in each row is the
# canonical recorded value (identical to the main tables); the bracket is the
# 1,000-iteration test-set bootstrap from data/bootstrap_metrics.json (which
# refits the probe, so its own point may differ by <0.01 -- every recorded
# point falls inside the reported interval). Pools mirror the main best-cell
# tables exactly (classification by macro-F1, regression by R^2).
# ===================================================================
def _ci(pair, dp=3):
    return f"[{f(pair[0], dp)}, {f(pair[1], dp)}]"


def build_headline_ci_cls():
    bc = BOOT["classification"]
    # (display, pool_tex, point_f1, point_kappa, boot_key)
    rows = [("Shuffled labels", "---", CLS_SHUF["test_macro_f1"],
             CLS_SHUF["test_kappa"], "shuffled")]
    for rid, disp in MAIN_COMPOSITION:
        r = CLS[rid]
        rows.append((disp, "---", r["test_macro_f1"], r["test_kappa"], rid))
    rule1 = len(rows)  # \midrule after shuffled + composition
    for enc in ENCODERS:
        pool, r = cls_best_pool(enc)
        rows.append((ENC_DISPLAY[enc], tt(pool), r["test_macro_f1"],
                     r["test_kappa"], f"{enc}_{pool}"))
    rule2 = len(rows)  # \midrule before ESM-2 upper bound
    e = CLS["esm2_650m"]
    rows.append(("ESM-2 650M", "---", e["test_macro_f1"], e["test_kappa"], "esm2_650m"))
    out = []
    for i, (disp, pool, f1, kap, key) in enumerate(rows):
        f1c = f"{f(f1, 3)} {_ci(bc[key]['macro_f1_ci95'])}"
        kc = f"{f(kap, 3)} {_ci(bc[key]['kappa_ci95'])}"
        out.append(f"{disp} & {pool} & {f1c} & {kc} \\\\")
        if i + 1 in (rule1, rule2):
            out.append(r"\midrule")
    return "\n".join(out)


def build_headline_ci_reg():
    br = BOOT["regression"]
    rows = []
    for rid, disp in MAIN_COMPOSITION:
        rows.append((disp, "---", REG[rid]["test_r2_macro"], rid))
    rule1 = len(rows)
    for enc in ENCODERS:
        pool, r = reg_best_pool(enc)
        rows.append((ENC_DISPLAY[enc], tt(pool), r["test_r2_macro"], f"{enc}_{pool}"))
    rule2 = len(rows)
    rows.append(("ESM-2 650M", "---", REG["esm2_650m"]["test_r2_macro"], "esm2_650m"))
    out = []
    for i, (disp, pool, r2, key) in enumerate(rows):
        r2c = f"{f(r2, 3)} {_ci(br[key]['r2_macro_ci95'])}"
        out.append(f"{disp} & {pool} & {r2c} \\\\")
        if i + 1 in (rule1, rule2):
            out.append(r"\midrule")
    return "\n".join(out)


# ===================================================================
# Table (sec 3.5): protein-LM comparison (main text, 3 dp)
# ESM-2 vs the best DNA encoder vs the composition floor, both readouts.
# Columns: Source & Macro-F1 & kappa & GenePT R^2.
# ===================================================================
def build_protein_comparison():
    k4, r4 = CLS[CDS_4MER], REG[CDS_4MER]
    aa_cls = max(["aa1", "aa2", "aa3"], key=lambda s: CLS[s]["test_macro_f1"])
    aa_reg = max(["aa1", "aa2", "aa3"], key=lambda s: REG[s]["test_r2_macro"])
    dna_cls = max(ENCODERS, key=lambda e: cls_best_pool(e)[1]["test_macro_f1"])
    dna_reg = max(ENCODERS, key=lambda e: reg_best_pool(e)[1]["test_r2_macro"])
    rows = [
        ("CDS 4-mer", k4["test_macro_f1"], k4["test_kappa"], r4["test_r2_macro"]),
        ("AA composition", CLS[aa_cls]["test_macro_f1"], CLS[aa_cls]["test_kappa"],
         REG[aa_reg]["test_r2_macro"]),
        ("Best DNA encoder", cls_best_pool(dna_cls)[1]["test_macro_f1"],
         cls_best_pool(dna_cls)[1]["test_kappa"], reg_best_pool(dna_reg)[1]["test_r2_macro"]),
        ("ESM-2 650M", CLS["esm2_650m"]["test_macro_f1"], CLS["esm2_650m"]["test_kappa"],
         REG["esm2_650m"]["test_r2_macro"]),
    ]
    bf1 = max(r[1] for r in rows); bk = max(r[2] for r in rows); br = max(r[3] for r in rows)
    out = []
    for lbl, f1, kp, r2 in rows:
        f1t = bold(f(f1, 3)) if f1 == bf1 else f(f1, 3)
        kt = bold(f(kp, 3)) if kp == bk else f(kp, 3)
        rt = bold(f(r2, 3)) if r2 == br else f(r2, 3)
        out.append(f"{lbl} & {f1t} & {kt} & {rt} \\\\")
    return "\n".join(out)


# ===================================================================
# Table (sec 3.x): homology leakage -- random vs homology split (main text)
# Best-pool macro-F1 per encoder, CDS and TSS, both splits + delta.
# (kappa is incomplete in the random metrics; macro-F1 is on both splits.)
# ===================================================================
def _best_f1_family5(metrics, enc, tss=False):
    cells = []
    for r in metrics:
        if r.get("task") != "family5" or r.get("shuffled_labels"):
            continue
        fs = r["feature_source"]
        is_tss = fs.startswith("tss_")
        if tss != is_tss:
            continue
        core = fs[4:] if is_tss else fs
        if core == enc or core.startswith(enc + "_"):
            cells.append(r)
    return max(cells, key=lambda r: r["test_macro_f1"])["test_macro_f1"] if cells else None


def _kmer_f1(metrics, *names):
    for fs in names:
        c = [r for r in metrics if r.get("task") == "family5" and r["feature_source"] == fs]
        if c:
            return c[0]["test_macro_f1"]
    return None


def build_leakage():
    out = []

    def section(label, baseline_rand, baseline_hom, tss):
        out.append(r"\multicolumn{4}{@{}l}{\textbf{" + label + r"}}\\")
        rows = [("4-mer", baseline_rand, baseline_hom)]
        for enc in ENCODERS:
            rows.append((ENC_DISPLAY[enc], _best_f1_family5(RAND, enc, tss),
                         _best_f1_family5(M, enc, tss)))
        for name, rv, hv in rows:
            out.append(f"\\quad {name} & {f(rv,3)} & {f(hv,3)} & {sgn(hv-rv,3)} \\\\")

    # random CDS 4-mer from the comparator re-run (RANDC, 0.662) to match the
    # split-comparison table; RAND's legacy 4-mer (0.672) differs by protocol.
    section(r"Coding sequence (CDS)", _kmer_f1(RANDC, "kmer"), _kmer_f1(M, "kmer"), False)
    out.append(r"\midrule")
    section(r"TSS-centred window (196{,}608\,bp)",
            _kmer_f1(RAND, "enformer_tss_4mer", "tss_kmer"),
            _kmer_f1(M, "enformer_tss_4mer", "tss_kmer"), True)
    return "\n".join(out)


# ===================================================================
# Random vs homology split comparison (CDS), classification + regression.
# DNA encoders/4-mer come from metrics.json; composition + ESM from
# metrics_random_comparators.json; homology from metrics_homology.json.
# ===================================================================
def _cls_f1(metrics, src):
    if src in ENCODERS:
        cells = [r for r in metrics if r.get("task") == "family5" and not r.get("shuffled_labels")
                 and not r["feature_source"].startswith("tss_")
                 and (r["feature_source"] == src or r["feature_source"].startswith(src + "_"))]
        return max(cells, key=lambda r: r["test_macro_f1"])["test_macro_f1"] if cells else None
    cells = [r for r in metrics if r.get("task") == "family5" and not r.get("shuffled_labels")
             and r.get("feature_source") == src]
    return cells[0]["test_macro_f1"] if cells else None


def _reg_r2(metrics, src):
    recs = [r for r in metrics if r.get("task") is None]
    if src in ENCODERS:
        cells = [r for r in recs if r.get("model") == "linear_probe"
                 and not str(r.get("dataset", "")).startswith("dataset_tss_")
                 and (str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "") == src
                      or str(r.get("dataset", "")).replace("dataset_", "").replace(".parquet", "").startswith(src + "_"))]
        return max(cells, key=lambda r: r["test_r2_macro"])["test_r2_macro"] if cells else None
    for r in recs:
        try:
            if reg_raw(r) == src:
                return r["test_r2_macro"]
        except Exception:
            continue
    return None


def _reg_r2_tss(metrics, enc):
    """Best-pool Ridge R^2 for an encoder on the TSS window."""
    cells = [r for r in metrics if r.get("task") is None and r.get("model") == "linear_probe"
             and str(r.get("dataset", "")).startswith("dataset_tss_")
             and (str(r.get("dataset", "")).replace("dataset_tss_", "").replace(".parquet", "") == enc
                  or str(r.get("dataset", "")).replace("dataset_tss_", "").replace(".parquet", "").startswith(enc + "_"))]
    return max(cells, key=lambda r: r["test_r2_macro"])["test_r2_macro"] if cells else None


def _rand_cls(src):
    v = _cls_f1(RANDC, src)
    return v if v is not None else _cls_f1(RAND, src)


def _rand_reg(src):
    v = _reg_r2(RANDC, src)
    return v if v is not None else _reg_r2(RAND, src)


CMP_DISPLAY = {"kmer": "CDS 4-mer", "kmer6": "CDS 6-mer", "codon": "Codon", "gc": "GC + length",
               "aa1": "AA 1-mer", "aa2": "AA 2-mer", "aa3": "AA 3-mer",
               "dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA",
               "esm2_150m": "ESM-2 150M", "esm2_650m": "ESM-2 650M"}
CMP_HEADLINE = ["kmer", "aa2", "aa3", "codon", "nt_v2", "dnabert2", "esm2_650m"]
CMP_FULL = ["kmer", "kmer6", "codon", "aa1", "aa2", "aa3",
            "dnabert2", "nt_v2", "gena_lm", "hyena_dna", "esm2_650m"]


def _split_comparison(srcs):
    out = []
    for src in srcs:
        rf, hf, rr, hr = _rand_cls(src), _cls_f1(M, src), _rand_reg(src), _reg_r2(M, src)

        def pair(rv, hv):
            return (f(rv, 3) + " & " + f(hv, 3)) if (rv is not None and hv is not None) else "--- & ---"

        out.append(f"{CMP_DISPLAY[src]} & {pair(rf, hf)} & {pair(rr, hr)} \\\\")
    return "\n".join(out)


# CDS section: composition / all four DNA encoders / protein-LM reference,
# separated by rules (no per-class labels) under one "Coding sequence" header.
SPLIT_CDS = [
    ["kmer", "codon", "aa2", "aa3"],
    ["nt_v2", "dnabert2", "gena_lm", "hyena_dna"],
    ["esm2_650m"],
]


def _enf_split(metrics, kind):
    """Enformer \\texttt{trunk\\_center} cell (same variant as Table~3); kind in {'f1','r2'}."""
    var = "enformer_trunk_center"
    if kind == "f1":
        c = [r for r in metrics if r.get("task") == "family5" and r.get("feature_source") == var]
        return c[0]["test_macro_f1"] if c else None
    c = [r for r in metrics if r.get("task") is None and var in str(r.get("dataset", ""))]
    return c[0]["test_r2_macro"] if c else None


def build_split_comparison():
    # Grouped rows with explicit leakage deltas (hom - rand). CDS classes from the
    # user spec, plus a TSS-window section so the homology-driven TSS collapse
    # (cited in the abstract/discussion) is shown rather than only asserted.
    def trip(rv, hv):
        return (f"{f(rv,3)} & {f(hv,3)} & {sgn(hv-rv,3)}"
                if rv is not None and hv is not None else "--- & --- & ---")
    out = [r"\multicolumn{7}{@{}l}{\textbf{Coding sequence (CDS)}}\\"]
    for bi, block in enumerate(SPLIT_CDS):
        if bi:
            out.append(r"\midrule")
        for src in block:
            f1 = trip(_rand_cls(src), _cls_f1(M, src))
            r2 = trip(_rand_reg(src), _reg_r2(M, src))
            out.append(f"\\quad {CMP_DISPLAY[src]} & {f1} & {r2} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{7}{@{}l}{\textbf{TSS window (196{,}608\,bp)}}\\")
    out.append(f"\\quad TSS 4-mer & {trip(_kmer_f1(RAND, 'enformer_tss_4mer', 'tss_kmer'), _kmer_f1(M, 'enformer_tss_4mer', 'tss_kmer'))}"
               f" & {trip(_reg_r2(RAND, 'enformer_tss_4mer'), _reg_r2(M, 'enformer_tss_4mer'))} \\\\")
    out.append(r"\midrule")
    for enc in ENCODERS:
        out.append(f"\\quad {ENC_DISPLAY[enc]} & {trip(_best_f1_family5(RAND, enc, True), _best_f1_family5(M, enc, True))}"
                   f" & {trip(_reg_r2_tss(RAND, enc), _reg_r2_tss(M, enc))} \\\\")
    out.append(r"\midrule")
    out.append(f"\\quad Enformer & {trip(_enf_split(RAND, 'f1'), _enf_split(ENFH, 'f1'))}"
               f" & {trip(_enf_split(RAND, 'r2'), _enf_split(ENFH, 'r2'))} \\\\")
    return "\n".join(out)


def build_split_comparison_full():
    return _split_comparison(CMP_FULL)


# ===================================================================
# Random-split mirrors of the homology appendix matrices (A1/A2).
# Same layout, computed on the random-stratified split. Classification
# reports macro-F1 (random DNA-LM runs have no cached kappa); regression
# mirrors the homology grid exactly.
# ===================================================================
def build_pooling_full_random():
    out = []
    base_f1 = CLS_RAND["kmer"]["test_macro_f1"]
    out.append(r"\multicolumn{5}{l}{\textbf{Coding sequence (CDS)}}\\")

    def row(disp, pool, r):
        return (f"{disp} & {pool} & {f(r['test_macro_f1'])} "
                f"& {sgn(r['test_macro_f1'] - base_f1)} & {f(r['test_accuracy'])} \\\\")

    for rid, disp in COMPOSITION:
        if rid in CLS_RAND:
            out.append(row(disp, "baseline" if rid == "kmer" else "---", CLS_RAND[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"{enc}_{pool}"
            if fsrc in CLS_RAND:
                out.append(row(ENC_DISPLAY[enc], tt(pool), CLS_RAND[fsrc]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{5}{l}{\textbf{Protein language model (translated CDS)}}\\")
    for rid, disp in ESM:
        if rid in CLS_RAND:
            out.append(row(disp, "---", CLS_RAND[rid]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{5}{l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    tbf = CLS_RAND[TSS_4MER]["test_macro_f1"]

    def trow(disp, pool, r):
        return (f"{disp} & {pool} & {f(r['test_macro_f1'])} "
                f"& {sgn(r['test_macro_f1'] - tbf)} & {f(r['test_accuracy'])} \\\\")

    out.append(trow("TSS 4-mer", "baseline", CLS_RAND[TSS_4MER]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"tss_{enc}_{pool}"
            if fsrc in CLS_RAND:
                out.append(trow(ENC_DISPLAY[enc], tt(pool), CLS_RAND[fsrc]))
    return "\n".join(out)


def build_regression_full_random():
    out = []
    base_r2 = REG_RAND["kmer"]["test_r2_macro"]
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Coding sequence (CDS)}}\\")

    def row(disp, pool, r, delta=True):
        d = sgn(r["test_r2_macro"] - base_r2) if delta else "---"
        cos = r.get("test_mean_cosine")
        cos = f(cos) if cos is not None else "---"
        return (f"{disp} & {pool} & {f(r['test_r2_macro'])} & {d} "
                f"& {cos} & {alpha_str(r.get('alpha'))} \\\\")

    for rid, disp in COMPOSITION:
        if rid in REG_RAND:
            out.append(row(disp, "baseline" if rid == "kmer" else "---", REG_RAND[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"{enc}_{pool}"
            if s in REG_RAND:
                out.append(row(ENC_DISPLAY[enc], tt(pool), REG_RAND[s]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{Protein language model (translated CDS)}}\\")
    for rid, disp in ESM:
        if rid in REG_RAND:
            out.append(row(disp, "---", REG_RAND[rid]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"tss_{enc}_{pool}"
            if s in REG_RAND:
                out.append(row(ENC_DISPLAY[enc], tt(pool), REG_RAND[s], delta=False))
    return "\n".join(out)


# ===================================================================
# Combined full-width matrices (appendix longtables): the homology-aware
# split section stacked above its random-stratified mirror -- one big table
# per readout. Random DNA-LM runs have no cached kappa, shown as '---'.
# write_raw emits just the rows (+ section headers) for \input into a longtable.
# ===================================================================
def write_raw(key, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.tex").write_text(body.rstrip() + "\n")
    print(f"wrote {key}.tex (full longtable)")


def _cell_order():
    """Canonical (kind, ...) stream shared by both readouts: context sub-headers
    plus one entry per encoder-pooling / baseline cell, in display order. The same
    key indexes the homology (CLS/REG) and random (CLS_RAND/REG_RAND) tables."""
    yield ("ctx", "Coding sequence (CDS)")
    for rid, disp in COMPOSITION:
        yield ("row", disp, "---", rid)
    yield ("rule",)
    for enc in ENCODERS:
        for pool in POOLS:
            yield ("row", ENC_DISPLAY[enc], tt(pool), f"{enc}_{pool}")
    yield ("rule",)  # ESM-2 set off by a rule, like Table 6 (no sub-header)
    for rid, disp in ESM:
        yield ("row", disp, "---", rid)
    yield ("ctx", r"TSS-centred window (196{,}608\,bp)")
    yield ("row", "TSS 4-mer", "---", TSS_4MER)
    yield ("rule",)
    for enc in ENCODERS:
        for pool in POOLS:
            yield ("row", ENC_DISPLAY[enc], tt(pool), f"tss_{enc}_{pool}")
    yield ("rule",)  # supervised Enformer TSS comparator, set off like Table 6
    yield ("row", "Enformer", "---", "enformer_trunk_center")


def _side_by_side(homidx, randidx, mcells):
    """Body rows for a side-by-side matrix: Enc|Pool|<hom metrics> | Enc|Pool|<rand metrics>."""
    out, first = [], True
    for item in _cell_order():
        kind = item[0]
        if kind == "ctx":
            if not first:
                out.append(r"\midrule")
            out.append(r"\multicolumn{10}{l}{\textbf{" + item[1] + r"}}\\")
            first = False
        elif kind == "rule":
            out.append(r"\midrule")
        else:
            _, disp, pool, key = item
            h, r = homidx.get(key), randidx.get(key)
            if h is None and r is None:
                continue
            out.append(f"{disp} & {pool} & {mcells(h)} & {disp} & {pool} & {mcells(r)} \\\\")
    return "\n".join(out)


def _side_longtable(caption, label, names, body):
    """Complete full-width page-breaking longtable with two side-by-side 5-column
    blocks (homology | random). Generated whole so the \\input sits at top level."""
    hdr = (r"\multicolumn{5}{c}{\textbf{Homology-aware split}} & "
           r"\multicolumn{5}{c}{\textbf{Random-stratified split}} \\" + "\n"
           + names + " & " + names + r" \\")
    return "\n".join([
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}llrrr@{\hspace{0.9em}\vrule width 1.1pt\hspace{0.9em}}llrrr@{}}",
        r"\caption{" + caption + r"\label{" + label + r"}}\\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{10}{l}{\emph{\tablename~\thetable\ -- continued}}\\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{10}{r}{\emph{continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        body,
        r"\end{longtable}}",
    ])


def _pool_section(idx):
    def crow(disp, pool, r):
        k = r.get("test_kappa")
        kt = f(k) if k is not None else "---"
        return f"{disp} & {pool} & {f(r['test_macro_f1'])} & {kt} & {f(r['test_accuracy'])} \\\\"
    o = [r"\multicolumn{5}{l}{\textit{Coding sequence (CDS)}}\\"]
    for rid, disp in COMPOSITION:
        if rid in idx:
            o.append(crow(disp, "---", idx[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"{enc}_{pool}"
            if fsrc in idx:
                o.append(crow(ENC_DISPLAY[enc], tt(pool), idx[fsrc]))
    if any(rid in idx for rid, _ in ESM):
        o.append(r"\midrule")
        o.append(r"\multicolumn{5}{l}{\textit{Protein language model (translated CDS)}}\\")
        for rid, disp in ESM:
            if rid in idx:
                o.append(crow(disp, "---", idx[rid]))
    o.append(r"\midrule")
    o.append(r"\multicolumn{5}{l}{\textit{TSS-centred window (196{,}608\,bp)}}\\")
    if TSS_4MER in idx:
        o.append(crow("TSS 4-mer", "---", idx[TSS_4MER]))
    for enc in ENCODERS:
        for pool in POOLS:
            fsrc = f"tss_{enc}_{pool}"
            if fsrc in idx:
                o.append(crow(ENC_DISPLAY[enc], tt(pool), idx[fsrc]))
    return o


def build_pooling_combined():
    def mc(rec):
        if rec is None:
            return "--- & --- & ---"
        k = rec.get("test_kappa")
        kt = f(k) if k is not None else "---"
        acc = rec.get("test_accuracy")
        at = f(acc) if acc is not None else "---"
        return f"{f(rec['test_macro_f1'])} & {kt} & {at}"
    enf = next((r for r in ENFH if r.get("task") == "family5"
                and r.get("feature_source") == "enformer_trunk_center"), None)
    return _side_longtable(
        r"Encoder-pooling cells for 5-way family classification, CDS and TSS: "
        r"homology-aware split (left) versus random-stratified split (right). "
        r"Random DNA-LM runs have no cached $\kappa$ (shown ``---''); the "
        r"supervised Enformer comparator (\texttt{trunk\_center}) closes the "
        r"TSS block.",
        "tab:s-pooling-full",
        r"Source & Pooling & Macro-F1 & $\kappa$ & Accuracy",
        _side_by_side({**CLS, "enformer_trunk_center": enf}, CLS_RAND, mc))


def _reg_section(idx):
    def rrow(disp, pool, r):
        cos = r.get("test_mean_cosine")
        cos = f(cos) if cos is not None else "---"
        a = r.get("alpha")
        at = alpha_str(a) if a is not None else "---"
        return f"{disp} & {pool} & {f(r['test_r2_macro'])} & {cos} & {at} \\\\"
    o = [r"\multicolumn{5}{l}{\textit{Coding sequence (CDS)}}\\"]
    for rid, disp in COMPOSITION:
        if rid in idx:
            o.append(rrow(disp, "---", idx[rid]))
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"{enc}_{pool}"
            if s in idx:
                o.append(rrow(ENC_DISPLAY[enc], tt(pool), idx[s]))
    if any(rid in idx for rid, _ in ESM):
        o.append(r"\midrule")
        o.append(r"\multicolumn{5}{l}{\textit{Protein language model (translated CDS)}}\\")
        for rid, disp in ESM:
            if rid in idx:
                o.append(rrow(disp, "---", idx[rid]))
    o.append(r"\midrule")
    o.append(r"\multicolumn{5}{l}{\textit{TSS-centred window (196{,}608\,bp)}}\\")
    if TSS_4MER in idx:
        o.append(rrow("TSS 4-mer", "---", idx[TSS_4MER]))
    for enc in ENCODERS:
        for pool in POOLS:
            s = f"tss_{enc}_{pool}"
            if s in idx:
                o.append(rrow(ENC_DISPLAY[enc], tt(pool), idx[s]))
    return o


def build_regression_combined():
    def mc(rec):
        if rec is None:
            return "--- & --- & ---"
        cos = rec.get("test_mean_cosine")
        cos = f(cos) if cos is not None else "---"
        a = rec.get("alpha")
        at = alpha_str(a) if a is not None else "---"
        return f"{f(rec['test_r2_macro'])} & {cos} & {at}"
    enf = next((r for r in ENFH if r.get("task") is None
                and "enformer_trunk_center" in (r.get("dataset") or "")), None)
    return _side_longtable(
        r"Ridge-to-GenePT cells, CDS and TSS: homology-aware split (left) "
        r"versus random-stratified split (right). The supervised Enformer "
        r"comparator (\texttt{trunk\_center}) closes the TSS block.",
        "tab:s-regression-full",
        r"Source & Pooling & $R^2$ macro & Mean cosine & $\alpha$",
        _side_by_side({**REG, "enformer_trunk_center": enf}, REG_RAND, mc))


# ===================================================================
# Supplementary (MLCB rebuttal): genomic-interval-disjoint TSS split (R2-Q4).
# Best-pool TSS macro-F1 on the primary homology split vs a split where TSS
# windows are also interval-disjoint. Removing window leakage pushes TSS toward
# the 0.224 chance floor -> the collapse conclusion is conservative.
# ===================================================================
def build_tss_disjoint():
    MTD = load("metrics_tss_disjoint.json")

    def t4(metrics):
        return [r for r in metrics if r.get("task") == "family5"
                and r["feature_source"] == TSS_4MER][0]["test_macro_f1"]

    out = [f"Shuffled labels (floor) & {f(CLS_SHUF['test_macro_f1'],3)} & --- & --- \\\\",
           f"TSS 4-mer & {f(t4(M),3)} & {f(t4(MTD),3)} & {sgn(t4(MTD)-t4(M),3)} \\\\",
           r"\midrule"]
    for enc in ENCODERS:
        h = _best_f1_family5(M, enc, tss=True)
        d = _best_f1_family5(MTD, enc, tss=True)
        out.append(f"{ENC_DISPLAY[enc]} & {f(h,3)} & {f(d,3)} & {sgn(d-h,3)} \\\\")
    return "\n".join(out)


# ===================================================================
# Supplementary (MLCB rebuttal): head-to-head paired-difference CIs (R1).
# Direct answer to "is 0.090 different from 0.060?" and the small-sample worry:
# every structural claim's paired CI excludes 0.
# ===================================================================
def build_paired_diff():
    pc, pr = BOOT["paired"]["classification"], BOOT["paired"]["regression"]
    out = [r"\multicolumn{3}{@{}l}{\textbf{Classification ($\Delta$Macro-F1)}}\\"]
    for disp, key in [
        ("NT-v2 (best) $-$ AA 2-mer", "nt_v2_meanD - aa2"),
        ("NT-v2 (best) $-$ 4-mer floor", "nt_v2_meanD - kmer"),
        ("ESM-2 650M $-$ AA 2-mer", "esm2_650m - aa2"),
        ("ESM-2 650M $-$ NT-v2 (best)", "esm2_650m - nt_v2_meanG"),
        ("ESM-2 650M $-$ ESM-2 150M", "esm2_650m - esm2_150m"),
    ]:
        c = pc[key]
        lo, hi = c["delta_macro_f1_ci95"]
        out.append(f"\\quad {disp} & {sgn(c['delta_macro_f1_point'],3)} "
                   f"[{sgn(lo,3)}, {sgn(hi,3)}] & {c['frac_A_gt_B_f1']:.3f} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{3}{@{}l}{\textbf{Regression ($\Delta$GenePT $R^2$)}}\\")
    for disp, key in [
        ("AA 3-mer $-$ AA 2-mer", "aa3 - aa2"),
        ("AA 3-mer $-$ 4-mer floor", "aa3 - kmer"),
        ("DNABERT-2 (best) $-$ AA 3-mer", "dnabert2_meanD - aa3"),
        ("ESM-2 650M $-$ AA 3-mer", "esm2_650m - aa3"),
        ("ESM-2 650M $-$ DNABERT-2 (best)", "esm2_650m - dnabert2_meanD"),
    ]:
        r = pr[key]
        lo, hi = r["delta_r2_macro_ci95"]
        out.append(f"\\quad {disp} & {sgn(r['delta_r2_macro_point'],3)} "
                   f"[{sgn(lo,3)}, {sgn(hi,3)}] & {r['frac_A_gt_B']:.3f} \\\\")
    return "\n".join(out)


# ===================================================================
# Supplementary (MLCB rebuttal): rotation-invariant Ridge metrics (R2-W4).
# macro-R^2 vs pooled (variance-weighted) R^2 vs coordinate-free retrieval@k;
# the method ordering is metric-invariant.
# ===================================================================
def build_ridge_robust():
    RR = load("ridge_robust.json")
    return "\n".join(
        f"{r['label']} & {f(r['macro_r2'],3)} & {f(r['pooled_r2'],3)} "
        f"& {r['top5']*100:.1f}\\% & {r['median_rank']:.0f} \\\\"
        for r in RR
    )


def main():
    write("family5_main", build_family5_main())
    write("ridge_main", build_ridge_main())
    write("cds_tss", build_cds_tss())
    write("split_comparison", build_split_comparison())
    write_raw("pooling_combined", build_pooling_combined())
    write_raw("regression_combined", build_regression_combined())
    write("s_seed_sensitivity", build_seed_sensitivity())
    write("s_cds_tss_paired", build_cds_tss_paired())
    write("s_headline_ci_cls", build_headline_ci_cls())
    write("s_headline_ci_reg", build_headline_ci_reg())
    write("s_tss_disjoint", build_tss_disjoint())
    write("s_paired_diff", build_paired_diff())
    write("s_ridge_robust", build_ridge_robust())
    print("\nAll fragments written to", OUT)


if __name__ == "__main__":
    main()
