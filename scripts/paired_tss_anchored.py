"""Paired bootstrap: TSS-anchored pooling vs each encoder's best global pool (MLCB E5, GAP 4).

``bootstrap_tss_anchored.py`` reports whether the anchored 95% CI lower bound clears a
*scalar* global-pool point. That is unpaired and ignores the global baseline's own
sampling uncertainty. The repo already uses a paired difference bootstrap everywhere
else in the revision (E2, the CDS-vs-TSS rows), so this runs the same paired test for
the anchored-vs-global comparison: resample the SAME test genes once per iteration and
apply to BOTH fixed predictions, giving Delta macro-F1 [CI] + P(anchored > global).

The anchored and best-pool TSS parquets are row-identical (3244 genes, same order), so
the pairing is exact; ``paired_bootstrap_classification`` intersects on ensembl_id
anyway. Anchored C = the confirmatory probe's per-cell best (``BEST_C`` below, copied
from ``bootstrap_tss_anchored.py``); global-pool C = the recorded best C read from the
split's metrics file (not hard-coded), so the paired point estimate reproduces the
known grid.

Best global pool per (encoder, split) — each encoder's hardest bar, matching
``center_chunk_finding.md`` and ``metrics_{homology,tss_disjoint}.json``:
  homology: dnabert2 meanmean, nt_v2 meanmean, gena_lm clsmean, hyena_dna meanD
  disjoint: dnabert2 meanmean, nt_v2 maxmean, gena_lm meanG,   hyena_dna meanG

No GPU. Never writes tracked data/metrics.json or data/splits.json (the disjoint pass
backs up + restores splits.json in a finally).

Run: uv run scripts/paired_tss_anchored.py
Writes: data/paired_tss_anchored.json
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
OUT = DATA / "paired_tss_anchored.json"
N_ITERS = 1000

# Best global pool per (split, encoder). C is looked up from the frozen paper metrics.
GLOBAL_POOL = {
    "homology": {"dnabert2": "meanmean", "nt_v2": "meanmean", "gena_lm": "clsmean", "hyena_dna": "meanD"},
    "disjoint": {"dnabert2": "meanmean", "nt_v2": "maxmean", "gena_lm": "meanG", "hyena_dna": "meanG"},
}
# Recorded best C. Anchored C is read from the anchored probe's own output (re-swept by
# the max_iter=5000 re-freeze); global-pool C from the frozen accepted-paper grid.
ANCHORED_METRICS = {
    "homology": DATA / "metrics_tss_anchored.json",
    "disjoint": DATA / "metrics_tss_anchored_disjoint.json",
}
POOL_METRICS = {
    "homology": DATA / "metrics_homology.json",
    "disjoint": DATA / "metrics_tss_disjoint.json",
}


def _C_from(metrics_path: Path, feature_source: str) -> float:
    rows = json.loads(metrics_path.read_text())
    for r in rows:
        if (r.get("task") == "family5" and not r.get("shuffled_labels")
                and r.get("feature_source") == feature_source):
            return float(r["C"])
    raise KeyError(f"{feature_source} not found in {metrics_path.name}")


def _register(split_label: str) -> None:
    for enc in ENCODERS:
        anc = DATA / f"dataset_tss_{enc}_tssanchored.parquet"
        pool = DATA / f"dataset_tss_{enc}_{GLOBAL_POOL[split_label][enc]}.parquet"
        for p in (anc, pool):
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")
        bt.DATASET_PATHS[f"tss_{enc}_tssanchored"] = anc
        bt.DATASET_PATHS[f"tss_{enc}_{GLOBAL_POOL[split_label][enc]}"] = pool


def _run_split(split_label: str) -> list[dict]:
    _register(split_label)
    rows = []
    for enc in ENCODERS:
        pool = GLOBAL_POOL[split_label][enc]
        c_anc = _C_from(ANCHORED_METRICS[split_label], f"tss_{enc}_tssanchored")
        c_pool = _C_from(POOL_METRICS[split_label], f"tss_{enc}_{pool}")
        res = bt.paired_bootstrap_classification(
            f"tss_{enc}_tssanchored", c_anc, f"tss_{enc}_{pool}", c_pool, n_iters=N_ITERS)
        lo, hi = res["delta_macro_f1_ci95"]
        rows.append({
            "encoder": enc, "split": split_label,
            "global_pool": pool, "C_anchored": c_anc, "C_global": c_pool,
            "delta_macro_f1_point": res["delta_macro_f1_point"],
            "delta_macro_f1_ci95": res["delta_macro_f1_ci95"],
            "frac_anchored_gt_global": res["frac_A_gt_B_f1"],
            "delta_kappa_point": res["delta_kappa_point"],
            "delta_kappa_ci95": res["delta_kappa_ci95"],
            "n_common": res["n_common"], "n_iters": N_ITERS,
        })
        favors = "anchored" if res["delta_macro_f1_point"] > 0 else "global"
        print(f"  [{split_label}] {enc:10} vs {pool:9} "
              f"dF1={res['delta_macro_f1_point']:+.4f} [{lo:+.3f},{hi:+.3f}] "
              f"P(anc>global)={res['frac_A_gt_B_f1']:.3f} n={res['n_common']} ({favors})", flush=True)
    return rows


def main() -> None:
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
