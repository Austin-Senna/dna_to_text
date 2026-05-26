"""Re-run the full probe matrix on the CURRENT split (data/splits.json).

Used for the journal revision's Phase 2: after switching splits.json to the
homology-aware split, re-run every classification + regression cell plus the
new compositional baselines, writing to a SEPARATE metrics file so the
random-split record (data/metrics.json) is preserved for comparison.

Cells whose parquet is absent in this checkout are skipped (reported). Each
cell reuses the tested CLI ``main()`` in-process via argv swapping, so the
recorded protocol matches single-cell runs exactly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import train_logistic_probe as tlp  # noqa: E402
import train_baseline as tb         # noqa: E402
import train_probe as tp            # noqa: E402

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
POOLINGS = ["meanmean", "specialmean", "maxmean", "clsmean", "meanD", "meanG"]
CLS_BASELINES = ["kmer", "kmer6", "codon", "aa1", "aa2", "aa3", "gc"]
REG_BASELINES = ["kmer", "kmer6", "codon", "aa1", "aa2", "aa3", "gc"]
CLS_TASKS = ["family5"]  # binary tasks added via --binary


def _exists_cls(dataset: str) -> bool:
    """Synthetic sources always runnable; parquet-backed need the file."""
    if dataset in tlp.SYNTHETIC_FEATURIZERS:
        return True
    p = tlp.DATASET_PATHS.get(dataset)
    return p is not None and Path(p).exists()


def _run(label: str, fn) -> bool:
    t0 = time.time()
    print(f"\n########## {label} ##########", flush=True)
    try:
        fn()
        print(f"########## OK {label} ({time.time()-t0:.1f}s) ##########", flush=True)
        return True
    except SystemExit:
        print(f"########## OK {label} ({time.time()-t0:.1f}s) ##########", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"########## FAIL {label}: {type(e).__name__}: {e} ##########", flush=True)
        return False


def _call(module, argv: list[str]):
    old = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", default=str(DATA / "metrics_homology.json"))
    ap.add_argument("--binary", action="store_true",
                    help="also run the two binary classification tasks")
    ap.add_argument("--tss", action="store_true",
                    help="also run TSS-arm cells (parquets must be present)")
    args = ap.parse_args()
    out = args.metrics_out

    tasks = list(CLS_TASKS)
    if args.binary:
        tasks += ["tf_vs_gpcr", "tf_vs_kinase"]

    cls_cells = list(CLS_BASELINES)
    cls_cells += [f"{e}_{p}" for e in ENCODERS for p in POOLINGS]
    if args.tss:
        cls_cells += [f"tss_{e}_{p}" for e in ENCODERS for p in POOLINGS]
        cls_cells += ["enformer_tss_4mer"]

    ok = fail = skip = 0
    print("=== CLASSIFICATION ===", flush=True)
    for task in tasks:
        for ds in cls_cells:
            if not _exists_cls(ds):
                print(f"  skip (no parquet): {ds} [{task}]", flush=True)
                skip += 1
                continue
            r = _run(f"cls {ds} [{task}]",
                     lambda ds=ds, task=task: _call(tlp, [
                         "train_logistic_probe.py", "--dataset", ds,
                         "--task", task, "--metrics-out", out]))
            ok += r
            fail += (not r)
    # anti-baseline (shuffled labels) on the headline encoder cell
    if _exists_cls("nt_v2_meanD"):
        r = _run("cls shuffled [family5]",
                 lambda: _call(tlp, [
                     "train_logistic_probe.py", "--dataset", "nt_v2_meanD",
                     "--task", "family5", "--shuffle-labels", "--metrics-out", out]))
        ok += r; fail += (not r)

    print("\n=== REGRESSION ===", flush=True)
    # baselines via train_baseline (--feature)
    for feat in REG_BASELINES:
        r = _run(f"reg baseline {feat}",
                 lambda feat=feat: _call(tb, [
                     "train_baseline.py", "--feature", feat, "--metrics-out", out]))
        ok += r; fail += (not r)
    # encoder cells via train_probe (--dataset PATH)
    reg_parquets = [f"dataset_{e}_{p}.parquet" for e in ENCODERS for p in POOLINGS]
    if args.tss:
        reg_parquets += [f"dataset_tss_{e}_{p}.parquet" for e in ENCODERS for p in POOLINGS]
    for fn in reg_parquets:
        path = DATA / fn
        if not path.exists():
            print(f"  skip (no parquet): {fn}", flush=True)
            skip += 1
            continue
        r = _run(f"reg {fn}",
                 lambda path=path: _call(tp, [
                     "train_probe.py", "--dataset", str(path),
                     "--probe-out", "/tmp/_phase2_probe.npz", "--metrics-out", out]))
        ok += r; fail += (not r)

    print(f"\n=== DONE: ok={ok} fail={fail} skip={skip} -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
