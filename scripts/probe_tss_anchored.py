"""Re-probe family5 on the TSS-anchored pooling (MLCB E5 confirmatory step).

Runs the same logistic family5 protocol that produced the centermean numbers
(``train_logistic_probe.main`` in-process via argv swap) on the TSS-anchored
datasets (``dataset_tss_<enc>_tssanchored.parquet``, built by
``build_tss_anchored_datasets.py``), on BOTH splits:

  - homology (current data/splits.json)  -> data/metrics_tss_anchored.json
  - genomic-interval-disjoint            -> data/metrics_tss_anchored_disjoint.json

The disjoint pass swaps in ``splits_tss_disjoint.json`` and restores the canonical
split AND any clobbered ``confusion_5way_*.json`` in a ``finally`` block (same safety
pattern as ``probe_tss_disjoint.py``). Never writes the tracked ``data/metrics.json``.
No GPU: TSS-anchored features are cached + split-independent.

Run: uv run scripts/probe_tss_anchored.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import train_logistic_probe as tlp  # noqa: E402
from splits.loader import SPLITS_PATH  # noqa: E402

ENCODERS = ["dnabert2", "nt_v2", "gena_lm", "hyena_dna"]
DISJOINT = DATA / "splits_tss_disjoint.json"
HOM_OUT = DATA / "metrics_tss_anchored.json"
DISJ_OUT = DATA / "metrics_tss_anchored_disjoint.json"


def _probe(enc: str, metrics_out: Path) -> None:
    old = sys.argv
    sys.argv = [
        "train_logistic_probe.py",
        "--dataset", f"tss_{enc}_tssanchored",
        "--task", "family5",
        "--metrics-out", str(metrics_out),
    ]
    try:
        tlp.main()
    finally:
        sys.argv = old


def _run_all(metrics_out: Path, label: str) -> None:
    print(f"\n===== {label}: probing {len(ENCODERS)} TSS-anchored datasets -> {metrics_out.name} =====",
          flush=True)
    if metrics_out.exists():
        metrics_out.unlink()
    for enc in ENCODERS:
        print(f"\n########## {label} tss_{enc}_tssanchored [family5] ##########", flush=True)
        _probe(enc, metrics_out)


def main() -> None:
    # 1) homology (current split)
    _run_all(HOM_OUT, "HOMOLOGY")

    # 2) disjoint split, with safe swap + restore
    if not DISJOINT.exists():
        raise SystemExit(f"missing {DISJOINT}; run make_tss_disjoint_split.py first")
    split_backup = SPLITS_PATH.read_bytes()
    conf_backup = {p: p.read_bytes() for p in DATA.glob("confusion_5way_*.json")}
    try:
        SPLITS_PATH.write_bytes(DISJOINT.read_bytes())
        print(f"\n=== swapped in {DISJOINT.name} ===", flush=True)
        _run_all(DISJ_OUT, "DISJOINT")
    finally:
        SPLITS_PATH.write_bytes(split_backup)
        for p, data in conf_backup.items():
            p.write_bytes(data)
        print(f"\nrestored canonical {SPLITS_PATH.name} and {len(conf_backup)} confusion matrices",
              flush=True)


if __name__ == "__main__":
    main()
