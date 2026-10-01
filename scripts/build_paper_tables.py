#!/usr/bin/env python3
"""Generate LaTeX table-body fragments for the dna_to_text_paper manuscript
from the camera-ready records.

The manuscript keeps its own ``\\begin{table}`` / caption / source-note
wrappers; this script writes each self-contained tabular (and the two appendix
longtables) that the paper ``\\input``s. Re-run to refresh every number; nothing
is hand-transcribed.

    uv run scripts/build_paper_tables.py

Inputs: ``data/v2`` through ``linear_trainer.records`` (one commit, the policy
purge, G7/G2) and ``data/v2/statistics.json`` (intervals, paired tests, null
bands; ``scripts/build_statistics.py``). Primary splits: the 40% homology split
for CDS, the genomic-interval-disjoint split for TSS. Every pick (pool, C or
alpha, nucleotide and amino-acid k, the best encoder) is made on validation
scores only; test metrics are reported, never ranked. A missing cell raises.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_loader.model_registry import encoder_pools
from data_loader.pool_names import POOL_DISPLAY, display_label
from linear_trainer import records as R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dna_to_text_paper" / "paper" / "tables"

CDS, TSS, RAND = "splits.json", "splits_tss_disjoint.json", "splits_random.json"
SEEDS = (1, 7, 123)
ENCODERS = list(R.ENCODERS)
ENC_DISPLAY = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
NT_DISPLAY = {"kmer": "CDS 4-mer", "kmer6": "CDS 6-mer"}
AA = [("aa1", "AA 1-mer"), ("aa2", "AA 2-mer"), ("aa3", "AA 3-mer")]
COMPOSITION = [("kmer", "CDS 4-mer"), ("kmer6", "CDS 6-mer"), ("codon", "Codon"), *AA]
TSS_4MER = "enformer_tss_4mer"
ENF_WHOLE, ENF_CENTRE = "enformer_trunk_global", "enformer_trunk_center"
F1, K, ACC, R2, COS = "test_macro_f1", "test_kappa", "test_accuracy", "test_r2_macro", "test_mean_cosine"


# ---------- formatting ----------
def f(x, d=4):
    return f"{x:.{d}f}"


def sgn(x, d=4):
    return f"{x:+.{d}f}"


def pool_name(s):
    return POOL_DISPLAY[s] if s and s != "base" else "---"


def bold(s):
    return r"\textbf{" + s + "}"


def alpha_str(a):
    a = float(a)
    return str(int(a)) if a >= 1 else ("%g" % a)


def _ci(pair, dp=3):
    return f"[{f(pair[0], dp)}, {f(pair[1], dp)}]"


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
    "split_comparison": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7.5}{9}\selectfont",
                             width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lrrr|rrr@{}",
                             header=r"Source & F1 (rand) & F1 (prim.) & $\Delta$F1 & $R^2$ (rand) & $R^2$ (prim.) & $\Delta R^2$"),
    "s_seed_sensitivity": dict(setup="", width=r"0.9\columnwidth",
                               cols=r"@{\extracolsep{\fill}}lrr@{}",
                               header=r"Cell & Macro-F1 (4 splits) & GenePT $R^2$ (4 splits)"),
    "s_cds_tss_paired": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{6.5}{7.5}\selectfont",
                             width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lccr@{}",
                             header=r"Encoder & $\Delta$Macro-F1 [95\% CI] & $\Delta R^2$ [95\% CI] & $P^*(\textrm{CDS}{>}\textrm{TSS})$"),
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
                          header=r"Comparison & $\Delta$ [95\% CI] & $p$"),
    "s_tss_anchored": dict(setup=r"\setlength{\tabcolsep}{3pt}\fontsize{7.5}{9}\selectfont",
                           width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrcr@{}",
                           header=r"Model & Whole window & TSS-Anchored [95\% CI] & Chunk comp."),
    "s_d5_sensitivity": dict(setup=r"\setlength{\tabcolsep}{2pt}\fontsize{7}{8.4}\selectfont",
                             width=r"\columnwidth", cols=r"@{\extracolsep{\fill}}lcr@{}",
                             header=r"Comparison & $\Delta$ [95\% CI] & $p$"),
    "s_split_population": dict(setup=r"\setlength{\tabcolsep}{3pt}\fontsize{7.5}{9}\selectfont",
                               width=r"0.9\columnwidth", cols=r"@{\extracolsep{\fill}}lrrrrr@{}",
                               header=r"Partition & Genes & Singleton & Median & In $\geq$10 & ORs / GPCRs"),
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


def write_raw(key, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{key}.tex").write_text(body.rstrip() + "\n")
    print(f"wrote {key}.tex (full longtable)")


# ---------- records ----------
class Split:
    """The records of one split file, by arm and task."""

    def __init__(self, name: str):
        self.name = name
        self.recs = R.load(name)

    def cells(self, arm: str, task: str) -> dict[str, dict]:
        return R.cells(self.recs, arm, task)

    def cell(self, arm: str, task: str, src: str) -> dict:
        by = self.cells(arm, task)
        if src not in by:
            raise R.MissingRecord(f"{self.name}: no {arm}/{task} record for {src}")
        return by[src]

    def best_pool(self, enc: str, arm: str, task: str) -> str:
        return R.best_pool(self.cells(arm, task), enc, arm)

    def best(self, enc: str, arm: str, task: str) -> dict:
        return self.cell(arm, task, self.best_pool(enc, arm, task))

    def nt(self, task: str) -> str:
        return R.best_nt_kmer(self.cells("cds", task))

    def aa(self, task: str) -> str:
        return R.best_aa(self.cells("cds", task))

    def best_encoder(self, arm: str, task: str) -> str:
        return R.best_encoder(self.cells(arm, task), arm)


HOM = DIS = RND = None                  # set by load_records()
SEED_SPLITS: dict[int, tuple[Split, Split]] = {}
STATS: dict = {}


def load_records() -> None:
    """Read every records file the tables use, and the statistics built from them."""
    global HOM, DIS, RND, STATS
    HOM, DIS, RND = Split(CDS), Split(TSS), Split(RAND)
    SEED_SPLITS.clear()
    SEED_SPLITS.update({s: (Split(f"splits_seed{s}.json"), Split(f"splits_tss_disjoint_seed{s}.json"))
                        for s in SEEDS})
    STATS = json.loads((R.V2 / "statistics.json").read_text())
    R.check_inputs(STATS)                 # no records file rewritten since statistics.json
    if STATS["n_iters"] != 1000:          # the captions say 1,000 resamples
        raise ValueError(f"statistics.json has {STATS['n_iters']} resamples, the captions say 1,000")
    # G7: every file the tables combine, and the statistics, from one commit and protocol.
    if STATS["stamp"] != R.stamp_of(HOM.recs, DIS.recs, RND.recs,
                                    *(s.recs for pair in SEED_SPLITS.values() for s in pair)):
        raise R.MixedRecords("statistics.json was built from other records than the tables read")


def interval(split: str, task: str, src: str) -> dict:
    return STATS["intervals"][f"{split}/{task}/{src}"]


def band(split: str, task: str, src: str) -> dict:
    return STATS["null_bands"][f"{split}/{task}/{src}"]


def pool_of(src: str) -> str:
    return src.rsplit("_", 1)[1]


def src_label(src: str) -> str:
    """'NT-v2 (Ends + Mean)'-style label for an encoder_pool source id."""
    return f"{ENC_DISPLAY[R.encoder_of(src)]} ({POOL_DISPLAY[pool_of(src)]})"


def null_row_cls(split: str) -> tuple[str, dict]:
    b = band(split, "family5", "kmer" if split == CDS else TSS_4MER)
    return "Shuffled labels (median of 200)", b


# ===================================================================
# Table 1: best 5-way family classification cell (main text, 3 dp)
# ===================================================================
def render_main_rows(rows, dp, rule_after=None):
    # rows: (display, pool, f1, df1, kappa, acc, is_control); None prints '---'
    cols = ("f1", "df1", "kappa", "acc")
    best = {c: max(m[c] for m in (dict(zip(cols, r[2:6])) for r in rows if not r[6])) for c in cols}
    out = []
    for i, (disp, pool, *vals, ctrl) in enumerate(rows):
        m = dict(zip(cols, vals))
        cells = []
        for c in cols:
            if m[c] is None:
                cells.append("---")
                continue
            txt = sgn(m[c], dp) if c == "df1" else f(m[c], dp)
            cells.append(bold(txt) if (not ctrl and m[c] == best[c]) else txt)
        out.append(f"{disp} & {pool} & " + " & ".join(cells) + r" \\")
        if rule_after is not None and i == rule_after - 1:
            out.append(r"\midrule")
    return "\n".join(out)


def _composition_rows(task: str) -> list[tuple[str, str]]:
    nt = HOM.nt(task)
    return [(nt, NT_DISPLAY[nt]), ("codon", "Codon"), *AA]


def build_family5_main():
    base = HOM.cell("cds", "family5", HOM.nt("family5"))[F1]
    disp, b = null_row_cls(CDS)
    rows = [(disp, "---", b["median"], b["median"] - base, b["medians"].get(K), b["medians"].get(ACC), True)]
    comp = _composition_rows("family5")
    for rid, d in comp:
        r = HOM.cell("cds", "family5", rid)
        rows.append((d, "---", r[F1], r[F1] - base, r[K], r[ACC], False))
    for enc in ENCODERS:
        src = HOM.best_pool(enc, "cds", "family5")
        r = HOM.cell("cds", "family5", src)
        rows.append((ENC_DISPLAY[enc], pool_name(pool_of(src)), r[F1], r[F1] - base, r[K], r[ACC], False))
    body = render_main_rows(rows, dp=3, rule_after=1 + len(comp))
    esm = HOM.cell("cds", "family5", "esm2_650m")
    esm_row = (f"ESM-2 650M & --- & {f(esm[F1], 3)} & {sgn(esm[F1] - base, 3)} "
               f"& {f(esm[K], 3)} & {f(esm[ACC], 3)} \\\\")
    return body + "\n" + r"\midrule" + "\n" + esm_row


# ===================================================================
# Table 2: best Ridge-to-GenePT cell (main text, 3 dp)
# ===================================================================
def build_ridge_main():
    base = HOM.cell("cds", "genept", HOM.nt("genept"))[R2]
    rows = []
    comp = _composition_rows("genept")
    for rid, d in comp:
        r = HOM.cell("cds", "genept", rid)
        rows.append((d, "---", r[R2], r[R2] - base, r[COS]))
    for enc in ENCODERS:
        src = HOM.best_pool(enc, "cds", "genept")
        r = HOM.cell("cds", "genept", src)
        rows.append((ENC_DISPLAY[enc], pool_name(pool_of(src)), r[R2], r[R2] - base, r[COS]))
    best_r2, best_cos = max(r[2] for r in rows), max(r[4] for r in rows)
    out = []
    for i, (disp, pool, r2, delta, cos) in enumerate(rows):
        r2t = bold(f(r2, 3)) if r2 == best_r2 else f(r2, 3)
        cost = bold(f(cos, 3)) if cos == best_cos else f(cos, 3)
        out.append(f"{disp} & {pool} & {r2t} & {sgn(delta, 3)} & {cost} \\\\")
        if i == len(comp) - 1:
            out.append(r"\midrule")
    esm = HOM.cell("cds", "genept", "esm2_650m")
    esm_row = f"ESM-2 650M & --- & {f(esm[R2], 3)} & {sgn(esm[R2] - base, 3)} & {f(esm[COS], 3)} \\\\"
    return "\n".join(out) + "\n" + r"\midrule" + "\n" + esm_row


# ===================================================================
# Table 3: substrate ablation CDS vs TSS (main text). CDS rows on the CDS
# primary (homology) split, TSS rows on the TSS primary (disjoint) split; Δ is
# within each block against its own 4-mer. Enformer: its whole-window mean.
# ===================================================================
def build_cds_tss():
    out = [r"\multicolumn{6}{@{}l}{\textbf{Coding sequence (CDS)}}\\"]
    nt_c, nt_r = HOM.cell("cds", "family5", "kmer"), HOM.cell("cds", "genept", "kmer")
    out.append(f"\\quad 4-mer & {f(nt_c[F1],3)} & {sgn(0,3)} & {f(nt_c[K],3)} & {f(nt_r[R2],3)} & {sgn(0,3)} \\\\")
    rows = []
    for enc in ENCODERS:
        c, r = HOM.best(enc, "cds", "family5"), HOM.best(enc, "cds", "genept")
        rows.append((enc, c[F1], c[F1] - nt_c[F1], c[K], r[R2], r[R2] - nt_r[R2]))
    bf, br = max(r[1] for r in rows), max(r[4] for r in rows)
    for enc, f1, df1, k, r2, dr2 in rows:
        out.append(f"\\quad {ENC_DISPLAY[enc]} & {bold(f(f1,3)) if f1 == bf else f(f1,3)} "
                   f"& {bold(sgn(df1,3)) if f1 == bf else sgn(df1,3)} & {f(k,3)} "
                   f"& {bold(f(r2,3)) if r2 == br else f(r2,3)} & {bold(sgn(dr2,3)) if r2 == br else sgn(dr2,3)} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\textbf{TSS-centred window (196{,}608\,bp)}}\\")
    t_c, t_r = DIS.cell("tss", "family5", TSS_4MER), DIS.cell("tss", "genept", TSS_4MER)
    out.append(f"\\quad 4-mer & {f(t_c[F1],3)} & {sgn(0,3)} & {f(t_c[K],3)} & {f(t_r[R2],3)} & {sgn(0,3)} \\\\")
    rows = []
    for enc in ENCODERS:
        c, r = DIS.best(enc, "tss", "family5"), DIS.best(enc, "tss", "genept")
        rows.append((enc, c[F1], c[F1] - t_c[F1], c[K], r[R2], r[R2] - t_r[R2]))
    bf = max(r[1] for r in rows)
    for enc, f1, df1, k, r2, dr2 in rows:
        out.append(f"\\quad {ENC_DISPLAY[enc]} & {bold(f(f1,3)) if f1 == bf else f(f1,3)} "
                   f"& {bold(sgn(df1,3)) if f1 == bf else sgn(df1,3)} & {f(k,3)} & {f(r2,3)} & {sgn(dr2,3)} \\\\")
    e_c, e_r = DIS.cell("tss", "family5", ENF_WHOLE), DIS.cell("tss", "genept", ENF_WHOLE)
    out.append(f"\\quad Enformer (whole window) & {f(e_c[F1],3)} & {sgn(e_c[F1]-t_c[F1],3)} & {f(e_c[K],3)} "
               f"& {f(e_r[R2],3)} & {sgn(e_r[R2]-t_r[R2],3)} \\\\")
    return "\n".join(out)


# ===================================================================
# Random vs homology split comparison: each split re-selects its own pools
# and k on its own validation set (CDS and TSS both random vs homology).
# ===================================================================
CMP_DISPLAY = {"codon": "Codon", "aa2": "AA 2-mer", "aa3": "AA 3-mer", "esm2_650m": "ESM-2 650M",
               **ENC_DISPLAY}
SPLIT_CDS = [["nt", "codon", "aa2", "aa3"], ["nt_v2", "dnabert2", "gena_lm", "hyena_dna"], ["esm2_650m"]]


def _value(split: Split, arm: str, task: str, name: str) -> float:
    m = F1 if task == "family5" else R2
    if name in ENCODERS:
        return split.best(name, arm, task)[m]
    if name == "nt":
        return split.cell(arm, task, split.nt(task))[m]
    return split.cell(arm, task, name)[m]


def _nt_label() -> str:
    """'CDS 4-mer' when every split and task picks the same k, else the picks."""
    picks = {(name, t): sp.nt(t) for name, sp in (("rand", RND), ("hom", HOM))
             for t in ("family5", "genept")}
    if len(set(picks.values())) == 1:
        return NT_DISPLAY[next(iter(picks.values()))]
    k = {v: v.removeprefix("kmer") or "4" for v in picks.values()}
    return "CDS k-mer (" + ", ".join(f"{n} {t[0].upper()}{k[v]}" for (n, t), v in picks.items()) + ")"


def primary(arm: str) -> Split:
    """The arm's primary split: homology for CDS, disjoint for TSS. On the homology
    split TSS windows overlap across partitions, unmasked by design (G25)."""
    return {"cds": HOM, "tss": DIS}[arm]


def build_split_comparison():
    def trip(rv, hv):
        return f"{f(rv,3)} & {f(hv,3)} & {sgn(hv-rv,3)}"

    out = [r"\multicolumn{7}{@{}l}{\textbf{Coding sequence (CDS)}}\\"]
    for bi, block in enumerate(SPLIT_CDS):
        if bi:
            out.append(r"\midrule")
        for src in block:
            name = _nt_label() if src == "nt" else CMP_DISPLAY[src]
            out.append(f"\\quad {name} & "
                       f"{trip(_value(RND, 'cds', 'family5', src), _value(primary('cds'), 'cds', 'family5', src))} & "
                       f"{trip(_value(RND, 'cds', 'genept', src), _value(primary('cds'), 'cds', 'genept', src))} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{7}{@{}l}{\textbf{TSS window (196{,}608\,bp; primary = genomic-interval-disjoint split)}}\\")
    for src, name in [(TSS_4MER, "TSS 4-mer"), *((e, ENC_DISPLAY[e]) for e in ENCODERS),
                      (ENF_WHOLE, "Enformer (whole window)")]:
        if src == ENCODERS[0] or src == ENF_WHOLE:
            out.append(r"\midrule")
        out.append(f"\\quad {name} & "
                   f"{trip(_value(RND, 'tss', 'family5', src), _value(primary('tss'), 'tss', 'family5', src))} & "
                   f"{trip(_value(RND, 'tss', 'genept', src), _value(primary('tss'), 'tss', 'genept', src))} \\\\")
    return "\n".join(out)


# ===================================================================
# Appendix longtables: every cell, primary split (left) beside the random
# split (right). CDS on the homology split, TSS on the disjoint split.
# ===================================================================
def _cell_order():
    yield ("ctx", "Coding sequence (CDS; primary = homology-aware split)", "cds")
    for rid, disp in COMPOSITION:
        yield ("row", disp, "---", rid, "cds")
    yield ("rule",)
    for enc in ENCODERS:
        for pool in encoder_pools(enc, "CDS"):
            yield ("row", ENC_DISPLAY[enc], pool_name(pool), f"{enc}_{pool}", "cds")
    yield ("rule",)
    yield ("row", "ESM-2 650M", "---", "esm2_650m", "cds")
    yield ("ctx", r"TSS-centred window (196{,}608\,bp; primary = genomic-interval-disjoint split)", "tss")
    yield ("row", "TSS 4-mer", "---", TSS_4MER, "tss")
    yield ("rule",)
    for enc in ENCODERS:
        for pool in encoder_pools(enc, "TSS"):
            yield ("row", ENC_DISPLAY[enc], pool_name(pool), f"tss_{enc}_{pool}", "tss")
    yield ("rule",)
    yield ("row", "Enformer (whole window)", "---", ENF_WHOLE, "tss")


def _side_by_side(task, mcells):
    out, first = [], True
    for item in _cell_order():
        if item[0] == "ctx":
            if not first:
                out.append(r"\midrule")
            out.append(r"\multicolumn{8}{l}{\textbf{" + item[1] + r"}}\\")
            first = False
        elif item[0] == "rule":
            out.append(r"\midrule")
        else:
            _, disp, pool, key, arm = item
            primary = HOM if arm == "cds" else DIS
            h, r = primary.cell(arm, task, key), RND.cell(arm, task, key)
            out.append(f"{disp} & {pool} & {mcells(h)} & {mcells(r)} \\\\")
    return "\n".join(out)


def _side_longtable(caption, label, metrics, body):
    hdr = (r" & & \multicolumn{3}{c}{\textbf{Primary split}} & "
           r"\multicolumn{3}{c}{\textbf{Random-stratified split}} \\" + "\n"
           + "Source & Pooling & " + metrics + " & " + metrics + r" \\")
    return "\n".join([
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{4pt}\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}llrrr@{\hspace{0.9em}\vrule width 1.1pt\hspace{0.9em}}rrr@{}}",
        r"\caption{" + caption + r"\label{" + label + r"}}\\",
        r"\toprule", hdr, r"\midrule", r"\endfirsthead",
        r"\multicolumn{8}{l}{\emph{\tablename~\thetable\ -- continued}}\\",
        r"\toprule", hdr, r"\midrule", r"\endhead",
        r"\midrule \multicolumn{8}{r}{\emph{continued on next page}}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
        body,
        r"\end{longtable}}",
    ])


def build_pooling_combined():
    return _side_longtable(
        r"Every cell for 5-way family classification: the primary split (left; CDS on the "
        r"homology-aware split, TSS on the genomic-interval-disjoint split) versus the "
        r"random-stratified split (right). Each split selects $C$ on its own validation set.",
        "tab:s-pooling-full", r"Macro-F1 & $\kappa$ & Accuracy",
        _side_by_side("family5", lambda r: f"{f(r[F1])} & {f(r[K])} & {f(r[ACC])}"))


def build_regression_combined():
    return _side_longtable(
        r"Every Ridge-to-GenePT cell: the primary split (left; CDS on the homology-aware "
        r"split, TSS on the genomic-interval-disjoint split) versus the random-stratified "
        r"split (right). Each split selects $\alpha$ on its own validation set.",
        "tab:s-regression-full", r"$R^2$ macro & Mean cosine & $\alpha$",
        _side_by_side("genept", lambda r: f"{f(r[R2])} & {f(r[COS])} & {alpha_str(r['alpha'])}"))


# ===================================================================
# Split-seed sensitivity: the primary split plus three re-seeded cluster
# assignments; each re-selects its own pools and k. Range over the four.
# ===================================================================
def build_seed_sensitivity():
    def ranges(get):
        vals = [get(HOM, DIS)] + [get(c, t) for c, t in SEED_SPLITS.values()]
        return f"${f(min(vals),3)}$--${f(max(vals),3)}$"     # math minus: "-0.013---0.010" drops one

    def best_dna(split, task):
        return split.cell("cds", task, split.best_encoder("cds", task))[F1 if task == "family5" else R2]

    def best_aa(split, task):
        return split.cell("cds", task, split.aa(task))[F1 if task == "family5" else R2]

    rows = [
        ("ESM-2 650M", lambda c, t: c.cell("cds", "family5", "esm2_650m")[F1],
         lambda c, t: c.cell("cds", "genept", "esm2_650m")[R2]),
        ("Best DNA encoder", lambda c, t: best_dna(c, "family5"), lambda c, t: best_dna(c, "genept")),
        ("AA composition", lambda c, t: best_aa(c, "family5"), lambda c, t: best_aa(c, "genept")),
        ("TSS (DNABERT-2)", lambda c, t: t.best("dnabert2", "tss", "family5")[F1],
         lambda c, t: t.best("dnabert2", "tss", "genept")[R2]),
    ]
    return "\n".join(f"{disp} & {ranges(fc)} & {ranges(fr)} \\\\" for disp, fc, fr in rows)


# ===================================================================
# CDS-vs-TSS paired cluster-bootstrap CIs (disjoint split: same test genes).
# P* is the share of resamples with CDS > TSS (not a posterior probability).
# ===================================================================
def build_cds_tss_paired():
    ex = STATS["exploratory"]
    rows = [(ENC_DISPLAY[e], f"{e} CDS > TSS") for e in ENCODERS]
    rows.append(("4-mer (control)", "kmer CDS > TSS 4-mer (control)"))
    out = []
    for disp, key in rows:
        c, r = ex[f"{TSS} family5: {key}"], ex[f"{TSS} genept: {key}"]
        f1 = f"{sgn(c['delta_point'],3)} [{sgn(c['delta_ci95'][0],3)}, {sgn(c['delta_ci95'][1],3)}]"
        r2 = f"{sgn(r['delta_point'],3)} [{sgn(r['delta_ci95'][0],3)}, {sgn(r['delta_ci95'][1],3)}]"
        out.append(f"{disp} & {f1} & {r2} & {c['p_a_gt_b']:.3f} \\\\")
    return "\n".join(out)


# ===================================================================
# Headline cluster-bootstrap CIs. The point is the recorded value; the
# bracket is the 1,000-resample cluster bootstrap of the stored predictions.
# ===================================================================
def build_headline_ci_cls():
    _, b = null_row_cls(CDS)
    rows = [f"Shuffled labels (median, null 2.5--97.5\\%) & --- & {f(b['median'],3)} {_ci(b['band95'])} "
            f"& --- \\\\", r"\midrule"]

    def row(d, pool, src):
        c = interval(CDS, "family5", src)
        return f"{d} & {pool} & {f(c['point'],3)} {_ci(c['ci95'])} & {f(c['kappa_point'],3)} {_ci(c['kappa_ci95'])} \\\\"

    for rid, d in _composition_rows("family5"):
        rows.append(row(d, "---", rid))
    rows.append(r"\midrule")
    for enc in ENCODERS:
        src = HOM.best_pool(enc, "cds", "family5")
        rows.append(row(ENC_DISPLAY[enc], pool_name(pool_of(src)), src))
    rows.append(r"\midrule")
    rows.append(row("ESM-2 650M", "---", "esm2_650m"))
    return "\n".join(rows)


def build_headline_ci_reg():
    rows = []

    def row(d, pool, src):
        c = interval(CDS, "genept", src)
        # 4 dp: the AA 3-mer vs AA 2-mer gap was quoted to R1 at this precision.
        return f"{d} & {pool} & {f(c['point'],4)} {_ci(c['ci95'], 4)} \\\\"

    for rid, d in _composition_rows("genept"):
        rows.append(row(d, "---", rid))
    rows.append(r"\midrule")
    for enc in ENCODERS:
        src = HOM.best_pool(enc, "cds", "genept")
        rows.append(row(ENC_DISPLAY[enc], pool_name(pool_of(src)), src))
    rows.append(r"\midrule")
    rows.append(row("ESM-2 650M", "---", "esm2_650m"))
    return "\n".join(rows)


# ===================================================================
# TSS on the homology split (window overlap allowed) vs the disjoint split.
# Each split re-selects its pools. The floor is each split's null band median.
# ===================================================================
def build_tss_disjoint():
    # The TSS null band runs on the disjoint split only (decided Oct 1).
    db = band(TSS, "family5", TSS_4MER)
    out = [f"Shuffled labels (median of 200) & --- & {f(db['median'],3)} & --- \\\\"]
    for src, name in [(TSS_4MER, "TSS 4-mer")]:
        h, d = HOM.cell("tss", "family5", src)[F1], DIS.cell("tss", "family5", src)[F1]
        out.append(f"{name} & {f(h,3)} & {f(d,3)} & {sgn(d-h,3)} \\\\")
    out.append(r"\midrule")
    for enc in ENCODERS:
        h, d = HOM.best(enc, "tss", "family5")[F1], DIS.best(enc, "tss", "family5")[F1]
        out.append(f"{ENC_DISPLAY[enc]} & {f(h,3)} & {f(d,3)} & {sgn(d-h,3)} \\\\")
    out.append(r"\midrule")
    for src, disp in ((ENF_WHOLE, "Enformer (whole window)"), (ENF_CENTRE, "Enformer (central 2{,}048\\,bp)")):
        h, d = HOM.cell("tss", "family5", src)[F1], DIS.cell("tss", "family5", src)[F1]
        out.append(f"{disp} & {f(h,3)} & {f(d,3)} & {sgn(d-h,3)} \\\\")
    return "\n".join(out)


# ===================================================================
# E5: TSS-Anchored vs whole-window pooling, both splits. The anchored cell
# carries its cluster-bootstrap CI, bold when the paired anchored - whole-window
# interval excludes 0 (caption: "Bold: the paired difference's 95% CI excludes
# 0"); the last column is the validation-selected composition
# (4-mer+GC or 6-mer) of the same anchored chunk.
# ===================================================================
def _beats(split_name: str, label: str, a: dict, b: dict) -> bool:
    """Bold a cell only when the paired test's interval for A - B excludes 0
    (G26); one cell's interval against the other's point ignores B's noise."""
    d = STATS["exploratory"][f"{split_name} family5: {label}"]
    if (d["a"], d["b"]) != (a["key"], b["key"]):
        raise R.MixedRecords(f"{label}: the paired test compares {d['a']} and {d['b']}, "
                             f"the table shows {a['key']} and {b['key']}")
    return d["delta_ci95"][0] > 0


def build_tss_anchored():
    out = []
    for split, name, title in ((HOM, CDS, "Homology-aware split"),
                               (DIS, TSS, "Genomic-interval-disjoint split")):
        if out:
            out.append(r"\midrule")
        out.append(r"\multicolumn{4}{@{}l}{\textbf{" + title + r"}}\\")
        by = split.cells("tss", "family5")
        for enc in ENCODERS:
            whole_rec = split.best(enc, "tss", "family5")
            whole = whole_rec[F1]
            c = interval(name, "family5", f"tss_{enc}_tssanchored")
            comp = by[R.pick(by, [f"tss_{enc}_chunk4mergc", f"tss_{enc}_chunk6mer"])][F1]
            txt = f"{f(c['point'],3)} {_ci(c['ci95'])}"
            win = _beats(name, f"{enc} anchored > whole-window", by[f"tss_{enc}_tssanchored"], whole_rec)
            out.append(f"{ENC_DISPLAY[enc]} & {f(whole,3)} & {bold(txt) if win else txt} "
                       f"& {f(comp,3)} \\\\")
        g_rec = split.cell("tss", "family5", ENF_WHOLE)
        c = interval(name, "family5", ENF_CENTRE)
        txt = f"{f(c['point'],3)} {_ci(c['ci95'])}"
        win = _beats(name, "Enformer centre > whole", by[ENF_CENTRE], g_rec)
        out.append(f"Enformer & {f(g_rec[F1],3)} & {bold(txt) if win else txt} & --- \\\\")
    return "\n".join(out)


# ===================================================================
# Paired differences: the four confirmatory tests (Holm-adjusted p) and the
# exploratory comparisons (one-sided p, unadjusted).
# ===================================================================
def _diff_line(disp, d, p):
    lo, hi = d["delta_ci95"]
    return f"\\quad {disp} & {sgn(d['delta_point'],3)} [{sgn(lo,3)}, {sgn(hi,3)}] & {p:.3f} \\\\"


def _key_label(key):
    src = key.rsplit("/", 1)[1]
    if src.endswith("_len"):          # Rule 3 control: composition + CDS length
        return _key_label(key.removesuffix("_len")) + " + length"
    if src in dict(COMPOSITION) or src in NT_DISPLAY:
        return dict(COMPOSITION)[src]
    if src.startswith("esm2_"):
        return "ESM-2 " + src.split("_")[1].upper()
    return src_label(src.removeprefix("tss_")) + (" (TSS)" if src.startswith("tss_") else "")


def build_paired_diff():
    line, name = _diff_line, _key_label
    conf = STATS["confirmatory"]
    out = [r"\multicolumn{3}{@{}l}{\textbf{Confirmatory, macro-F1 (Holm-adjusted $p$)}}\\"]
    for k, d in conf.items():
        where = " (disjoint split)" if k.startswith("T4") else ""
        out.append(line(f"{name(d['a'])} $-$ {name(d['b'])}{where}", d, d["p_holm"]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{3}{@{}l}{\textbf{Exploratory, GenePT $R^2$ (unadjusted $p$)}}\\")
    ex = STATS["exploratory"]
    for k in ("aa3 > aa2", "aa_kmer > nt_kmer", "encoder > aa_kmer", "esm2_650m > aa_kmer",
              "esm2_650m > encoder"):
        d = ex[f"{CDS} genept: {k}"]
        out.append(line(f"{name(d['a'])} $-$ {name(d['b'])}", d, d["p_one_sided"]))
    return "\n".join(out)


# ===================================================================
# D5: Ends + Mean at matched regularisation, and the headline tests with each
# disclosed defect masked from scoring. Exploratory, unadjusted p.
# ===================================================================
def build_d5_sensitivity():
    out = [r"\multicolumn{3}{@{}l}{\textbf{Ends + Mean $-$ Mean $\times 3$ (Mean at $3\times C$), macro-F1 "
           r"(one-sided $p$, unadjusted)}}\\"]
    p3 = STATS["pooling_3x"]
    for e in ENCODERS:
        d = p3[f"family5/{e}"]["Ends+Mean > Mean x3 (3x C)"]
        out.append(_diff_line(ENC_DISPLAY[e], d, d["p_one_sided"]))
    out.append(r"\midrule")
    out.append(r"\multicolumn{3}{@{}l}{\textbf{Encoder $-$ composition with log CDS length, macro-F1 "
               r"(one-sided $p$, unadjusted)}}\\")
    for k in ("nt_kmer+len", "aa_kmer+len"):
        d = STATS["exploratory"][f"{CDS} family5: encoder > {k}"]
        out.append(_diff_line(f"{_key_label(d['a'])} $-$ {_key_label(d['b'])}", d, d["p_one_sided"]))
    for name, title, metric in (("label_noise", "noisy TF labels (non-C2H2 zinc-finger groups only", "macro-F1"),
                                ("label_noise_kinase", "kinase-labelled genes that are not protein kinases "
                                 "(HGNC kinase groups of scaffolds, subunits or small-molecule kinases",
                                 "macro-F1"),
                                ("template", "templated GenePT summaries (shared text", "GenePT $R^2$")):
        tests = STATS["sensitivity"][name]["tests"]
        if any(d.get("n_excluded", 0) == 0 for d in tests.values()):
            raise ValueError(f"{name}: a masked test excluded no scored genes; the mask is empty there")
        cds = {d["n_excluded"] for k, d in tests.items() if not k.startswith("T4")}
        dis = {d["n_excluded"] for k, d in tests.items() if k.startswith("T4")}
        if len(cds) != 1 or len(dis) > 1:
            raise ValueError(f"{name}: the masked tests exclude different numbers of genes: {cds}, {dis}")
        n = f"{cds.pop()} test genes removed" + (f", {dis.pop()} on the disjoint split" if dis else "")
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{Without {title}; {n}), {metric} (one-sided $p$, unadjusted)}}}}\\")
        for k, d in tests.items():
            where = " (disjoint split)" if k.startswith("T4") else ""
            out.append(_diff_line(f"{_key_label(d['a'])} $-$ {_key_label(d['b'])}{where}", d, d["p_one_sided"]))
    return "\n".join(out)


# ===================================================================
# Who the test sets are (Rule 3): cluster-size profile and olfactory receptors
# per partition, from data/v2/counts.json (scripts/build_counts.py).
# ===================================================================
def build_split_population():
    counts = json.loads((R.V2 / "counts.json").read_text())
    for name in (CDS, TSS):                       # built from the split files on disk now
        if counts["inputs"][name] != R._sha(R.REPO_ROOT / "data" / name):
            raise R.MixedRecords(f"counts.json was built from another {name}; rerun scripts/build_counts.py")
    pop = counts["split_population"]
    out = []
    for name, title in ((CDS, "Homology-aware split (CDS primary)"),
                        (TSS, "Genomic-interval-disjoint split (TSS primary)")):
        if out:
            out.append(r"\midrule")
        out.append(r"\multicolumn{6}{@{}l}{\textbf{" + title + r"}}\\")
        for part in ("train", "val", "test"):
            r = pop[name][part]
            out.append(f"\\quad {part.capitalize()} & {r['n']:,} & {r['singleton_share']*100:.0f}\\% "
                       f"& {r['median_cluster_size']:.0f} & {r['share_in_clusters_ge10']*100:.0f}\\% "
                       f"& {r['olfactory']} / {r['gpcr']} \\\\".replace(",", "{,}"))
    return "\n".join(out)


# ===================================================================
# Rotation-invariant Ridge metrics (R2-W4), rescored from stored predictions.
# ===================================================================
def build_ridge_robust():
    rr = json.loads((R.V2 / "ridge_robust.json").read_text())
    R.check_inputs(rr)
    if rr["stamp"] != STATS["stamp"]:                                   # G7
        raise R.MixedRecords("ridge_robust.json was rescored from other records; re-run "
                             "scripts/ridge_robust_metrics.py")
    return "\n".join(
        f"{display_label(r['label'])} & {f(r['macro_r2'],3)} "
        f"& {f(r['pooled_r2'],3)} & {r['top5']*100:.1f}\\% & {r['median_rank']:.0f} \\\\"
        for r in rr["rows"])


def main():
    load_records()
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
    write("s_tss_anchored", build_tss_anchored())
    write("s_d5_sensitivity", build_d5_sensitivity())
    write("s_split_population", build_split_population())
    print("\nAll fragments written to", OUT)


if __name__ == "__main__":
    main()
