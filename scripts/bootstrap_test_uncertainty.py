"""Bootstrap test-set 95% CIs for the headline classification and regression cells.

For each headline cell (validation-selected, from headline_cells.py):
  1. Load the cell's stored test predictions from its record (``pred_file``),
     verified against ``pred_sha256``. Nothing is refitted: a refit can land on
     a different model (another machine, thread count or max_iter), and the
     bootstrap would then no longer bracket the table value (ledger G8).
  2. Check that the point estimate equals the record's test value.
  3. Bootstrap-resample the test set 1,000 iterations (stratified by family
     for classification; iid by gene for regression) and recompute the
     metric on each resample. Report 95% percentile CIs.

Also computes per-class F1 from the full test predictions for each
classification cell (no bootstrap needed; just the standard per-class F1).

Output: data/bootstrap_metrics.json (and a stdout summary table).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from linear_trainer import sources
from linear_trainer.cell import load_predictions
from linear_trainer.fit import score
from headline_cells import (BEST_DNA_CLS, BEST_DNA_REG, CLS_BEST, CLS_BEST_TSS, CLS_RECS,
                            ENCODERS, HOMOLOGY, REG_BEST, REG_BEST_TSS, REG_RECS)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "bootstrap_metrics.json"

# The shared registry (scripts that register extra parquets write into it).
DATASET_PATHS = sources.DATASET_PATHS

# Headline cells to bootstrap: (cell_name, record). Every cell is the
# validation-selected pool from headline_cells.py (same rule as the tables), so
# the bootstrapped point estimate is the value it brackets.
COMPARATORS = ("kmer", "codon", "aa1", "aa2", "aa3")
CLS_RECS = {**CLS_RECS, "tss_4mer": CLS_RECS["enformer_tss_4mer"]}
REG_RECS = {**REG_RECS, "tss_4mer": REG_RECS["enformer_tss_4mer"]}


def _cells(recs: dict[str, dict], names) -> list[tuple[str, dict]]:
    return [(n, recs[n]) for n in names]


_shuf = next(r for r in HOMOLOGY if r.get("task") == "family5" and r.get("shuffled_labels"))
HEADLINE_CLS = (_cells(CLS_RECS, COMPARATORS + tuple(CLS_BEST.values()) + ("esm2_650m",))
                + [("shuffled", _shuf)])
HEADLINE_REG = _cells(REG_RECS, COMPARATORS + tuple(REG_BEST.values()) + ("esm2_650m",))
HEADLINE_CLS_TSS = _cells(CLS_RECS, ("tss_4mer",) + tuple(CLS_BEST_TSS.values()))
HEADLINE_REG_TSS = _cells(REG_RECS, ("tss_4mer",) + tuple(REG_BEST_TSS.values()))


def _stored(rec: dict, kind: str, value_key: str) -> dict:
    """The record's stored predictions, checked to reproduce its test value exactly."""
    arrays = load_predictions(rec)
    point = score(kind, arrays["y_true"], arrays["pred"])[value_key]
    if point != rec[value_key]:
        raise RuntimeError(f"{rec.get('run_id', '?')}: stored predictions give {value_key}="
                           f"{point!r}, record says {rec[value_key]!r}")
    return arrays


def bootstrap_classification(rec: dict, n_iters: int = 1000, seed: int = 42) -> dict:
    arrays = _stored(rec, "logistic", "test_macro_f1")
    y_te, y_pred = arrays["y_true"], arrays["pred"]

    # Stratified bootstrap by true class
    classes = sorted(np.unique(y_te).tolist())
    rng = np.random.default_rng(seed)
    f1s, kappas = [], []
    for _ in range(n_iters):
        parts = []
        for c in classes:
            ci = np.where(y_te == c)[0]
            parts.append(rng.choice(ci, size=len(ci), replace=True))
        idx = np.concatenate(parts)
        f1s.append(f1_score(y_te[idx], y_pred[idx], average="macro"))
        kappas.append(cohen_kappa_score(y_te[idx], y_pred[idx]))

    per_class = f1_score(y_te, y_pred, average=None, labels=classes)
    return {
        "n_test": int(len(y_te)),
        "macro_f1_point": rec["test_macro_f1"],
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)),
                          float(np.percentile(f1s, 97.5))],
        "kappa_point": float(cohen_kappa_score(y_te, y_pred)),
        "kappa_ci95": [float(np.percentile(kappas, 2.5)),
                       float(np.percentile(kappas, 97.5))],
        "per_class_f1": {c: float(per_class[i]) for i, c in enumerate(classes)},
        "n_iters": n_iters,
    }


def _r2_macro(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)))


def bootstrap_regression(rec: dict, n_iters: int = 1000, seed: int = 42) -> dict:
    arrays = _stored(rec, "ridge", "test_r2_macro")
    Y_te, Y_pred = arrays["y_true"], arrays["pred"]

    rng = np.random.default_rng(seed)
    n = len(Y_te)
    r2s = [_r2_macro(Y_te[idx], Y_pred[idx])
           for idx in (rng.choice(n, size=n, replace=True) for _ in range(n_iters))]
    return {
        "n_test": int(n),
        "r2_macro_point": rec["test_r2_macro"],
        "r2_macro_ci95": [float(np.percentile(r2s, 2.5)),
                          float(np.percentile(r2s, 97.5))],
        "n_iters": n_iters,
    }


# ---------------------------------------------------------------------------
# Paired difference tests (issue #10): resample the SAME test genes once per
# iteration and apply to BOTH cells' stored predictions, so the bootstrap CI is
# on the metric *difference* rather than on two independent cells. Reports the
# difference CI plus the fraction of resamples favouring side A (a one-sided
# paired bootstrap p-value analogue). Cells whose test sets differ (CDS vs TSS
# on different split files) are paired on their common genes.
# ---------------------------------------------------------------------------

def _pair(recs: dict[str, dict], label: str, a: str, b: str):
    return (label, recs[a], recs[b])


PAIRED_CLS = [_pair(CLS_RECS, *x) for x in [
    (f"{BEST_DNA_CLS} - aa2", BEST_DNA_CLS, "aa2"),  # headline: best DNA-LM vs AA composition
    (f"{BEST_DNA_CLS} - kmer", BEST_DNA_CLS, "kmer"),
    ("aa2 - kmer", "aa2", "kmer"),
    # ESM-2 protein-LM comparator (#9): vs AA composition, vs best DNA-LM, and scaling.
    ("esm2_650m - aa2", "esm2_650m", "aa2"),
    (f"esm2_650m - {BEST_DNA_CLS}", "esm2_650m", BEST_DNA_CLS),
    ("esm2_650m - esm2_150m", "esm2_650m", "esm2_150m"),
    # CDS arm vs TSS arm within each DNA-LM encoder (#10).
    *[(f"{e} CDS - TSS", CLS_BEST[e], CLS_BEST_TSS[e]) for e in ENCODERS],
    ("kmer CDS - TSS 4mer", "kmer", "tss_4mer"),
]]

PAIRED_REG = [_pair(REG_RECS, *x) for x in [
    (f"{BEST_DNA_REG} - aa3", BEST_DNA_REG, "aa3"),  # headline reg: best DNA-LM vs AA-3mer
    (f"{BEST_DNA_REG} - kmer", BEST_DNA_REG, "kmer"),
    ("aa3 - kmer", "aa3", "kmer"),
    ("aa3 - aa2", "aa3", "aa2"),  # R1's example: 0.090 vs 0.060
    ("aa2 - aa1", "aa2", "aa1"),  # AA composition ladder is monotone
    # ESM-2 protein-LM comparator (#9): vs AA-3mer, vs best DNA-LM, and scaling.
    ("esm2_650m - aa3", "esm2_650m", "aa3"),
    (f"esm2_650m - {BEST_DNA_REG}", "esm2_650m", BEST_DNA_REG),
    ("esm2_650m - esm2_150m", "esm2_650m", "esm2_150m"),
    # CDS arm vs TSS arm within each DNA-LM encoder (#10).
    *[(f"{e} CDS - TSS", REG_BEST[e], REG_BEST_TSS[e]) for e in ENCODERS],
    ("kmer CDS - TSS 4mer", "kmer", "tss_4mer"),
]]


def _common_alignment(ids_a: np.ndarray, ids_b: np.ndarray):
    """Return index arrays (pos_a, pos_b) selecting the shared genes, in the
    order they appear in ids_a. Used to pair CDS- and TSS-arm test sets that
    may not cover identical gene sets."""
    pos_b = {g: i for i, g in enumerate(ids_b.tolist())}
    keep_a, keep_b = [], []
    for i, g in enumerate(ids_a.tolist()):
        j = pos_b.get(g)
        if j is not None:
            keep_a.append(i)
            keep_b.append(j)
    return np.array(keep_a, dtype=int), np.array(keep_b, dtype=int)


def paired_bootstrap_classification(rec_a: dict, rec_b: dict,
                                    n_iters: int = 1000, seed: int = 42) -> dict:
    a = _stored(rec_a, "logistic", "test_macro_f1")
    b = _stored(rec_b, "logistic", "test_macro_f1")
    pa, pb = _common_alignment(a["ids"], b["ids"])
    y = a["y_true"][pa]
    assert np.array_equal(y, b["y_true"][pb]), "paired test labels misaligned"
    pred_a, pred_b = a["pred"][pa], b["pred"][pb]

    classes = sorted(np.unique(y).tolist())
    rng = np.random.default_rng(seed)
    d_f1, d_kappa = [], []
    for _ in range(n_iters):
        idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                          size=int((y == c).sum()), replace=True)
                              for c in classes])
        d_f1.append(f1_score(y[idx], pred_a[idx], average="macro")
                    - f1_score(y[idx], pred_b[idx], average="macro"))
        d_kappa.append(cohen_kappa_score(y[idx], pred_a[idx])
                       - cohen_kappa_score(y[idx], pred_b[idx]))
    d_f1 = np.asarray(d_f1)
    d_kappa = np.asarray(d_kappa)
    return {
        "n_common": int(len(y)),
        "delta_macro_f1_point": float(
            f1_score(y, pred_a, average="macro") - f1_score(y, pred_b, average="macro")),
        "delta_macro_f1_ci95": [float(np.percentile(d_f1, 2.5)),
                                float(np.percentile(d_f1, 97.5))],
        "delta_kappa_point": float(
            cohen_kappa_score(y, pred_a) - cohen_kappa_score(y, pred_b)),
        "delta_kappa_ci95": [float(np.percentile(d_kappa, 2.5)),
                             float(np.percentile(d_kappa, 97.5))],
        "frac_A_gt_B_f1": float(np.mean(d_f1 > 0)),
        "n_iters": n_iters,
    }


def paired_bootstrap_regression(rec_a: dict, rec_b: dict,
                                n_iters: int = 1000, seed: int = 42) -> dict:
    a = _stored(rec_a, "ridge", "test_r2_macro")
    b = _stored(rec_b, "ridge", "test_r2_macro")
    pa, pb = _common_alignment(a["ids"], b["ids"])
    Y = a["y_true"][pa]
    assert np.allclose(Y, b["y_true"][pb]), "paired regression targets misaligned"
    pred_a, pred_b = a["pred"][pa], b["pred"][pb]

    rng = np.random.default_rng(seed)
    n = len(Y)
    d_r2 = []
    for _ in range(n_iters):
        idx = rng.choice(n, size=n, replace=True)
        d_r2.append(_r2_macro(Y[idx], pred_a[idx]) - _r2_macro(Y[idx], pred_b[idx]))
    d_r2 = np.asarray(d_r2)
    return {
        "n_common": int(n),
        "delta_r2_macro_point": float(_r2_macro(Y, pred_a) - _r2_macro(Y, pred_b)),
        "delta_r2_macro_ci95": [float(np.percentile(d_r2, 2.5)),
                                float(np.percentile(d_r2, 97.5))],
        "frac_A_gt_B": float(np.mean(d_r2 > 0)),
        "n_iters": n_iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--paired", action="store_true",
                    help="also compute paired difference CIs (issue #10)")
    args = ap.parse_args()

    results = {"classification": {}, "regression": {},
               "n_iters": args.n_iters, "seed": args.seed}

    print("=== Classification bootstrap CIs (family5) ===")
    for name, rec in HEADLINE_CLS + HEADLINE_CLS_TSS:
        t0 = time.time()
        res = bootstrap_classification(rec, n_iters=args.n_iters, seed=args.seed)
        results["classification"][name] = res
        f1_lo, f1_hi = res["macro_f1_ci95"]
        k_lo, k_hi = res["kappa_ci95"]
        per = res["per_class_f1"]
        print(f"  {name:<26s} F1={res['macro_f1_point']:.4f} [{f1_lo:.3f}-{f1_hi:.3f}]  "
              f"kappa={res['kappa_point']:.4f} [{k_lo:.3f}-{k_hi:.3f}]  "
              f"({time.time()-t0:.1f}s)")
        per_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(per.items()))
        print(f"    per-class F1: {per_str}")

    print("\n=== Regression bootstrap CIs (Ridge -> GenePT) ===")
    for name, rec in HEADLINE_REG + HEADLINE_REG_TSS:
        t0 = time.time()
        res = bootstrap_regression(rec, n_iters=args.n_iters, seed=args.seed)
        results["regression"][name] = res
        r2_lo, r2_hi = res["r2_macro_ci95"]
        print(f"  {name:<26s} R2={res['r2_macro_point']:.4f} [{r2_lo:.3f}-{r2_hi:.3f}]  "
              f"({time.time()-t0:.1f}s)")

    if args.paired:
        results["paired"] = {"classification": {}, "regression": {}}
        print("\n=== Paired classification difference CIs (A - B) ===")
        for label, ra, rb in PAIRED_CLS:
            t0 = time.time()
            res = paired_bootstrap_classification(ra, rb, n_iters=args.n_iters, seed=args.seed)
            results["paired"]["classification"][label] = res
            lo, hi = res["delta_macro_f1_ci95"]
            print(f"  {label:<24s} dF1={res['delta_macro_f1_point']:+.4f} [{lo:+.3f},{hi:+.3f}]  "
                  f"P(A>B)={res['frac_A_gt_B_f1']:.3f}  n={res['n_common']}  ({time.time()-t0:.1f}s)")

        print("\n=== Paired regression difference CIs (A - B) ===")
        for label, ra, rb in PAIRED_REG:
            t0 = time.time()
            res = paired_bootstrap_regression(ra, rb, n_iters=args.n_iters, seed=args.seed)
            results["paired"]["regression"][label] = res
            lo, hi = res["delta_r2_macro_ci95"]
            print(f"  {label:<24s} dR2={res['delta_r2_macro_point']:+.4f} [{lo:+.3f},{hi:+.3f}]  "
                  f"P(A>B)={res['frac_A_gt_B']:.3f}  n={res['n_common']}  ({time.time()-t0:.1f}s)")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
