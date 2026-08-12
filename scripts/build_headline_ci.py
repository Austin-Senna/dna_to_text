"""Regenerate the headline bootstrap-CI supplement tables at 4 decimals (MLCB R1).

R1 asked for extra decimal places in the supplement "to future-proof". The two CI
tables (``s_headline_ci_cls.tex``, ``s_headline_ci_reg.tex``) had lost their generator
(the headline cell list was later trimmed), so they were static. This recreates them,
data-driven: for each displayed cell it reads the recorded validation-selected
hyperparameter from ``metrics_homology.json`` and calls the paper's own bootstrap
functions (1,000 iters, seed 42), formatting at 4 dp. Nothing is hand-transcribed.

Reuses the display indexes/helpers in ``build_paper_tables`` (DRY). Reads the current
``data/splits.json`` (must be the canonical homology split).

Run: uv run scripts/build_headline_ci.py
"""
from __future__ import annotations

import build_paper_tables as T
from bootstrap_test_uncertainty import bootstrap_classification, bootstrap_regression

N_ITERS = 1000
SEED = 42


def _ci(point: float, lo_hi: list[float], d: int = 4) -> str:
    return f"{point:.{d}f} [{lo_hi[0]:.{d}f}, {lo_hi[1]:.{d}f}]"


def _wrap(setup: str, cols: str, header: str, body: str) -> str:
    return (
        "{" + setup + r"\begin{tabular*}{0.9\columnwidth}{" + cols + r"}\toprule" + "\n"
        + header + r" \\\midrule" + "\n"
        + body.rstrip() + "\n"
        + r"\botrule" + "\n"
        + r"\end{tabular*}}" + "\n"
    )


def build_cls() -> str:
    """Source | Pool | Macro-F1 [95% CI] | kappa [95% CI], at 4 dp."""
    def row(disp, pool, src, c, shuf):
        res = bootstrap_classification(src, float(c), shuf, n_iters=N_ITERS, seed=SEED)
        return (f"{disp} & {pool} & {_ci(res['macro_f1_point'], res['macro_f1_ci95'])} "
                f"& {_ci(res['kappa_point'], res['kappa_ci95'])} \\\\")

    comp = [row("Shuffled labels", "---", T.CLS_SHUF["feature_source"], T.CLS_SHUF["C"], True)]
    for rid, disp in [("kmer", "CDS 4-mer")] + T.COMPOSITION[2:]:  # 4-mer, codon, aa1-3
        comp.append(row(disp, "---", rid, T.CLS[rid]["C"], False))
    enc = [row(T.ENC_DISPLAY[e], T.tt(p), rec["feature_source"], rec["C"], False)
           for e in T.ENCODERS for p, rec in [T.cls_best_pool(e)]]
    esm = [row("ESM-2 650M", "---", "esm2_650m", T.CLS["esm2_650m"]["C"], False)]
    return "\n".join(comp + [r"\midrule"] + enc + [r"\midrule"] + esm)


def build_reg() -> str:
    """Source | Pool | GenePT R^2 [95% CI], at 4 dp."""
    def row(disp, pool, src, alpha):
        res = bootstrap_regression(src, float(alpha), False, n_iters=N_ITERS, seed=SEED)
        return f"{disp} & {pool} & {_ci(res['r2_macro_point'], res['r2_macro_ci95'])} \\\\"

    def reg_src(rec):
        return rec["dataset"].replace("dataset_", "").replace(".parquet", "")

    comp = [row(disp, "---", rid, T.REG[rid]["alpha"])
            for rid, disp in [("kmer", "CDS 4-mer")] + T.COMPOSITION[2:]]
    enc = [row(T.ENC_DISPLAY[e], T.tt(p), reg_src(rec), rec["alpha"])
           for e in T.ENCODERS for p, rec in [T.reg_best_pool(e)]]
    esm = [row("ESM-2 650M", "---", "esm2_650m", T.REG["esm2_650m"]["alpha"])]
    return "\n".join(comp + [r"\midrule"] + enc + [r"\midrule"] + esm)


def main() -> None:
    setup = r"\setlength{\tabcolsep}{3pt}\fontsize{7.5}{9}\selectfont"
    cls = _wrap(setup, r"@{\extracolsep{\fill}}llcc@{}",
                r"Source & Pool & Macro-F1 [95\% CI] & $\kappa$ [95\% CI]", build_cls())
    reg = _wrap(setup, r"@{\extracolsep{\fill}}llc@{}",
                r"Source & Pool & GenePT $R^2$ [95\% CI]", build_reg())
    T.OUT.mkdir(parents=True, exist_ok=True)
    (T.OUT / "s_headline_ci_cls.tex").write_text(cls)
    (T.OUT / "s_headline_ci_reg.tex").write_text(reg)
    print(f"wrote s_headline_ci_cls.tex ({cls.count(chr(10))} lines) and "
          f"s_headline_ci_reg.tex ({reg.count(chr(10))} lines) at 4 dp -> {T.OUT}")


if __name__ == "__main__":
    main()
