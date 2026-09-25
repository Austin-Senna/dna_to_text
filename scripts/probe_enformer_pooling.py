"""Matched-pooling Enformer comparison on both TSS splits (MLCB R2 point 3).

The paper compares Enformer's centred readout (``trunk_center``, central 16 x 128 bp
bins) against the DNA encoders' whole-window global pools, which mixes two pooling
regimes. This puts both on the same footing:

  Part 1, whole window:  each encoder's best global pool   vs Enformer ``trunk_global``
  Part 2, TSS-centred:   each encoder's TSS-anchored chunk vs Enformer ``trunk_center``

on the homology split and the genomic-interval-disjoint split. Enformer was probed only
on homology before (``probe_enformer_homology.py``, its own sklearn loop); here it goes
through the canonical ``train_logistic_probe`` protocol so every cell shares one fit.

Steps per split:
  1. probe Enformer global + center (C swept on val, refit train+val, test)
  2. stratified bootstrap 95% CI on each Enformer cell
  3. paired bootstrap: Enformer center - global, and each encoder vs Enformer within
     each part (same test genes resampled for both predictions)

The disjoint pass swaps in ``splits_tss_disjoint.json``; ``splits.json`` and every
``confusion_5way_*.json`` are restored in a ``finally`` block. Never writes the tracked
``data/metrics.json``. No GPU: all features are cached and split-independent.

Part 2 needs ``dataset_tss_<enc>_tssanchored.parquet`` (gitignored; built by
``build_tss_anchored_datasets.py`` from the GPU-extracted ``tss_chunk_reductions_<enc>/``
cache). ``--parts whole_window`` runs steps 1-2, the Enformer center-global test, and
Part 1 without it.

Run: uv run scripts/probe_enformer_pooling.py [--parts whole_window tss_centred]
Writes: data/metrics_enformer_pooling{,_disjoint}.json, data/enformer_pooling.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import bootstrap_test_uncertainty as bt  # noqa: E402
import train_logistic_probe as tlp  # noqa: E402
from paired_tss_anchored import GLOBAL_POOL, ANCHORED_METRICS, POOL_METRICS, _C_from  # noqa: E402
from splits.loader import SPLITS_PATH  # noqa: E402

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
ENFORMER = {"global": "enformer_trunk_global", "center": "enformer_trunk_center"}
DISJOINT = DATA / "splits_tss_disjoint.json"
METRICS_OUT = {
    "homology": DATA / "metrics_enformer_pooling.json",
    "disjoint": DATA / "metrics_enformer_pooling_disjoint.json",
}
OUT = DATA / "enformer_pooling.json"
N_ITERS = 1000


def _probe(dataset: str, metrics_out: Path) -> None:
    old = sys.argv
    sys.argv = ["train_logistic_probe.py", "--dataset", dataset,
                "--task", "family5", "--metrics-out", str(metrics_out)]
    try:
        tlp.main()
    finally:
        sys.argv = old


def _register(split_label: str, parts: list[str]) -> None:
    for ds in ENFORMER.values():
        bt.DATASET_PATHS[ds] = DATA / f"dataset_{ds}.parquet"
    for enc in ENCODERS:
        if "tss_centred" in parts:
            bt.DATASET_PATHS[f"tss_{enc}_tssanchored"] = DATA / f"dataset_tss_{enc}_tssanchored.parquet"
        pool = GLOBAL_POOL[split_label][enc]
        bt.DATASET_PATHS[f"tss_{enc}_{pool}"] = DATA / f"dataset_tss_{enc}_{pool}.parquet"


def _paired(split_label: str, part: str, ds_a: str, c_a: float, ds_b: str, c_b: float) -> dict:
    res = bt.paired_bootstrap_classification(ds_a, c_a, ds_b, c_b, n_iters=N_ITERS)
    lo, hi = res["delta_macro_f1_ci95"]
    print(f"  [{split_label}/{part}] {ds_a:28} - {ds_b:24} "
          f"dF1={res['delta_macro_f1_point']:+.4f} [{lo:+.3f},{hi:+.3f}] "
          f"P(A>B)={res['frac_A_gt_B_f1']:.3f}", flush=True)
    return {"split": split_label, "part": part, "a": ds_a, "C_a": c_a, "b": ds_b, "C_b": c_b,
            "delta_macro_f1_point": res["delta_macro_f1_point"],
            "delta_macro_f1_ci95": res["delta_macro_f1_ci95"],
            "frac_a_gt_b_f1": res["frac_A_gt_B_f1"],
            "n_common": res["n_common"], "n_iters": N_ITERS}


def _run_split(split_label: str, parts: list[str]) -> dict:
    metrics_out = METRICS_OUT[split_label]
    if metrics_out.exists():
        metrics_out.unlink()
    for ds in ENFORMER.values():
        print(f"\n##### {split_label}: probe {ds} #####", flush=True)
        _probe(ds, metrics_out)

    _register(split_label, parts)
    enf_C = {k: _C_from(metrics_out, ds) for k, ds in ENFORMER.items()}
    cells = {}
    for k, ds in ENFORMER.items():
        b = bt.bootstrap_classification(ds, enf_C[k], shuffled=False, n_iters=N_ITERS)
        cells[ds] = {"C": enf_C[k], "macro_f1_point": b["macro_f1_point"],
                     "macro_f1_ci95": b["macro_f1_ci95"], "per_class_f1": b["per_class_f1"]}
        lo, hi = b["macro_f1_ci95"]
        print(f"  [{split_label}] {ds:24} F1={b['macro_f1_point']:.4f} [{lo:.3f},{hi:.3f}] C={enf_C[k]}",
              flush=True)

    paired = [_paired(split_label, "enformer", ENFORMER["center"], enf_C["center"],
                      ENFORMER["global"], enf_C["global"])]
    for enc in ENCODERS if "whole_window" in parts else []:
        pool = GLOBAL_POOL[split_label][enc]
        c_pool = _C_from(POOL_METRICS[split_label], f"tss_{enc}_{pool}")
        paired.append(_paired(split_label, "whole_window", f"tss_{enc}_{pool}", c_pool,
                              ENFORMER["global"], enf_C["global"]))
    for enc in ENCODERS if "tss_centred" in parts else []:
        c_anc = _C_from(ANCHORED_METRICS[split_label], f"tss_{enc}_tssanchored")
        paired.append(_paired(split_label, "tss_centred", f"tss_{enc}_tssanchored", c_anc,
                              ENFORMER["center"], enf_C["center"]))
    return {"enformer_cells": cells, "paired": paired}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", choices=["whole_window", "tss_centred"],
                    default=["whole_window", "tss_centred"])
    args = ap.parse_args()
    if "tss_centred" in args.parts:
        missing = [p.name for enc in ENCODERS
                   if not (p := DATA / f"dataset_tss_{enc}_tssanchored.parquet").exists()]
        if missing:
            raise SystemExit(f"missing {missing}; build them with build_tss_anchored_datasets.py "
                             f"or pass --parts whole_window")
    if not DISJOINT.exists():
        raise SystemExit(f"missing {DISJOINT}; run make_tss_disjoint_split.py first")
    split_backup = SPLITS_PATH.read_bytes()
    conf_backup = {p: p.read_bytes() for p in DATA.glob("confusion_5way_*.json")}
    results = {}
    try:
        print("=== HOMOLOGY (current splits.json) ===", flush=True)
        results["homology"] = _run_split("homology", args.parts)
        SPLITS_PATH.write_bytes(DISJOINT.read_bytes())
        print(f"\n=== DISJOINT (swapped in {DISJOINT.name}) ===", flush=True)
        results["disjoint"] = _run_split("disjoint", args.parts)
    finally:
        SPLITS_PATH.write_bytes(split_backup)
        for p, data in conf_backup.items():
            p.write_bytes(data)
        print(f"\nrestored canonical {SPLITS_PATH.name} and {len(conf_backup)} confusion matrices",
              flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
