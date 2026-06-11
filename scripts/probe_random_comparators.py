"""Probe composition baselines + ESM-2 on the legacy random-stratified split (sensitivity).

The journal-revision comparators (kmer6/codon/aa1-3/gc + esm2_{150m,650m}) were added
during the hardening phase and only ever probed on the homology split. This re-runs them
on the original random 70/15/15 split (``data/splits_random.json``) under the same
``--select-by r2`` protocol used for ``data/metrics_homology.json``, so the
random-vs-homology sensitivity of the AA-composition-ties-DNA-LM and ESM-2-dominance
findings can be reported.

The canonical ``splits.json`` (currently the 40% homology split) is backed up and
restored in a ``finally`` block. Metrics land in ``data/metrics_random_comparators.json``
— the legacy ``data/metrics.json`` and the primary ``data/metrics_homology.json`` are
NOT touched. ESM-2 classification overwrites the tracked
``data/confusion_5way_esm2_*.json`` matrices — restore after the run with
``git checkout -- data/confusion_5way_esm2_*.json``.

Run: uv run scripts/probe_random_comparators.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import train_logistic_probe as tlp  # noqa: E402
import train_baseline as tb  # noqa: E402
import train_probe as tp  # noqa: E402

SPLITS = DATA / "splits.json"
RANDOM = DATA / "splits_random.json"
OUT = DATA / "metrics_random_comparators.json"

# Comparators added during the journal-hardening revision (homology-only until now).
CLS_CELLS = ["kmer", "kmer6", "codon", "aa1", "aa2", "aa3", "gc", "esm2_150m", "esm2_650m"]
REG_BASELINES = ["kmer", "kmer6", "codon", "aa1", "aa2", "aa3", "gc"]
REG_PARQUETS = ["dataset_esm2_150m.parquet", "dataset_esm2_650m.parquet"]


def _call(module, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = argv
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old


def _run_cells(out: Path) -> None:
    for ds in CLS_CELLS:
        runnable = ds in tlp.SYNTHETIC_FEATURIZERS or (
            tlp.DATASET_PATHS.get(ds) and Path(tlp.DATASET_PATHS[ds]).exists()
        )
        if not runnable:
            print(f"  skip cls {ds}: not resolvable", flush=True)
            continue
        _call(tlp, ["train_logistic_probe.py", "--dataset", ds, "--task", "family5",
                    "--metrics-out", str(out)])
    for feat in REG_BASELINES:
        _call(tb, ["train_baseline.py", "--feature", feat, "--metrics-out", str(out)])
    for fn in REG_PARQUETS:
        path = DATA / fn
        if not path.exists():
            print(f"  skip reg {fn}: parquet absent", flush=True)
            continue
        _call(tp, ["train_probe.py", "--dataset", str(path),
                   "--probe-out", "/tmp/_randomcmp_probe.npz", "--metrics-out", str(out)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=RANDOM)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.split.exists():
        raise SystemExit(f"missing split file: {args.split}")

    backup = SPLITS.read_bytes() if SPLITS.exists() else None
    try:
        SPLITS.write_bytes(args.split.read_bytes())
        print(f"=== swapped in {args.split.name} ===", flush=True)
        if args.out.exists():
            args.out.unlink()
        _run_cells(args.out)
    finally:
        if backup is not None:
            SPLITS.write_bytes(backup)
            print(f"\nrestored canonical {SPLITS.name}", flush=True)

    print("\nNOTE: restore clobbered ESM-2 confusion matrices: "
          "git checkout -- data/confusion_5way_esm2_*.json")


if __name__ == "__main__":
    main()
