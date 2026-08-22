"""Bootstrap 95% CIs for the TSS-anchored family5 probe (MLCB E5 finalize).

The confirmatory re-probe gave point estimates only; nt_v2's anchored number in
particular (disjoint 0.374 > homology 0.351, wrong direction) needs a CI before it
can be trusted. This puts a stratified test-set bootstrap CI on the anchored
macro-F1 for all 4 encoders on both splits, reusing the repo's existing
``bootstrap_test_uncertainty.bootstrap_classification`` (refit with the given C,
predict test, resample test 1000x stratified by family, 2.5/97.5 percentile).

A ``--seed`` re-run would be useless (lbfgs is deterministic on a fixed split);
the test-set bootstrap is the correct uncertainty source.

No GPU. Never writes tracked data/metrics.json or data/splits.json (the disjoint
pass backs up + restores splits.json in a finally).

Run: uv run scripts/bootstrap_tss_anchored.py
Writes: data/bootstrap_tss_anchored.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import bootstrap_test_uncertainty as bt  # noqa: E402
from splits.loader import SPLITS_PATH  # noqa: E402

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
DISJOINT = DATA / "splits_tss_disjoint.json"
OUT = DATA / "bootstrap_tss_anchored.json"
N_ITERS = 1000

# Anchored best C is READ from the probe's own metrics output (not hard-coded) so
# it always tracks the current sweep — the max_iter=5000 re-freeze re-sweeps C, and a
# stale hard-coded value would silently mismatch the frozen point estimate.
METRICS_FILE = {
    "homology": DATA / "metrics_tss_anchored.json",
    "disjoint": DATA / "metrics_tss_anchored_disjoint.json",
}
# Global-pool point estimates for the overlap check (family5 macro-F1). These are the
# FROZEN accepted-paper global-pool numbers (metrics_homology.json / metrics_tss_disjoint.json,
# computed at max_iter=2000) — the published bar the anchored CI must clear. NOT regenerated.
GLOBAL_POOL = {
    "homology": {"dnabert2": 0.326, "nt_v2": 0.313, "gena_lm": 0.244, "hyena_dna": 0.287},
    "disjoint": {"dnabert2": 0.254, "nt_v2": 0.259, "gena_lm": 0.206, "hyena_dna": 0.265},
}


def _anchored_C(split_label: str, enc: str) -> float:
    rows = json.loads(METRICS_FILE[split_label].read_text())
    fs = f"tss_{enc}_tssanchored"
    for r in rows:
        if (r.get("task") == "family5" and not r.get("shuffled_labels")
                and r.get("feature_source") == fs):
            return float(r["C"])
    raise KeyError(f"{fs} not in {METRICS_FILE[split_label].name}; run probe_tss_anchored.py first")


def _register_anchored() -> None:
    for enc in ENCODERS:
        p = DATA / f"dataset_tss_{enc}_tssanchored.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run build_tss_anchored_datasets.py first")
        bt.DATASET_PATHS[f"tss_{enc}_tssanchored"] = p


def _run_split(split_label: str) -> list[dict]:
    rows = []
    for enc in ENCODERS:
        C = _anchored_C(split_label, enc)
        res = bt.bootstrap_classification(f"tss_{enc}_tssanchored", C=C, shuffled=False, n_iters=N_ITERS)
        gp = GLOBAL_POOL[split_label][enc]
        lo, hi = res["macro_f1_ci95"]
        clears_global = lo > gp
        rows.append({
            "encoder": enc, "split": split_label, "C": C,
            "macro_f1_point": res["macro_f1_point"], "macro_f1_ci95": res["macro_f1_ci95"],
            "global_pool_point": gp, "ci_clears_global_pool": clears_global,
            "kappa_point": res["kappa_point"], "kappa_ci95": res["kappa_ci95"],
            "per_class_f1": res["per_class_f1"], "n_test": res["n_test"], "n_iters": N_ITERS,
        })
        print(f"  [{split_label}] {enc:10} F1={res['macro_f1_point']:.3f} "
              f"CI95=[{lo:.3f},{hi:.3f}] vs global {gp:.3f} "
              f"-> {'CLEARS' if clears_global else 'overlaps'} global", flush=True)
    return rows


def main() -> None:
    _register_anchored()
    results = []

    print("=== HOMOLOGY (current splits.json) ===", flush=True)
    results += _run_split("homology")

    if not DISJOINT.exists():
        raise SystemExit(f"missing {DISJOINT}; run make_tss_disjoint_split.py first")
    split_backup = SPLITS_PATH.read_bytes()
    try:
        SPLITS_PATH.write_bytes(DISJOINT.read_bytes())
        print("\n=== DISJOINT (swapped in splits_tss_disjoint.json) ===", flush=True)
        results += _run_split("disjoint")
    finally:
        SPLITS_PATH.write_bytes(split_backup)
        print(f"\nrestored canonical {SPLITS_PATH.name}", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
