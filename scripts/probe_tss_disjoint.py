"""Re-probe the TSS arm on the genomic-interval-disjoint split (MLCB R2-Q4).

Reviewer 2 noted the 196,608 bp TSS windows overlap across the protein-cluster split
(48.3% of test genes). ``make_tss_disjoint_split.py`` builds a split with zero
cross-split window overlap; this swaps it in (restoring the canonical split AND any
clobbered ``confusion_5way_*.json`` in a ``finally`` block) and re-probes ONLY the TSS
arm via ``rerun_on_split --only-tss``, writing to ``data/metrics_tss_disjoint.json`` so
``data/metrics_homology.json`` is untouched. The TSS embeddings are split-independent,
so no GPU re-extraction is needed.

Run: uv run scripts/probe_tss_disjoint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import rerun_on_split as ros  # noqa: E402
from splits.loader import SPLITS_PATH  # noqa: E402

DISJOINT = DATA / "splits_tss_disjoint.json"
OUT = DATA / "metrics_tss_disjoint.json"


def _call_main(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = argv
    try:
        ros.main()
    finally:
        sys.argv = old


def main() -> None:
    if not DISJOINT.exists():
        raise SystemExit(f"missing {DISJOINT}; run make_tss_disjoint_split.py first")

    split_backup = SPLITS_PATH.read_bytes()
    conf_backup = {p: p.read_bytes() for p in DATA.glob("confusion_5way_*.json")}
    try:
        SPLITS_PATH.write_bytes(DISJOINT.read_bytes())
        print(f"=== swapped in {DISJOINT.name} -> re-probing TSS arm ===", flush=True)
        if OUT.exists():
            OUT.unlink()
        _call_main(["rerun_on_split.py", "--only-tss", "--metrics-out", str(OUT)])
    finally:
        SPLITS_PATH.write_bytes(split_backup)
        for p, data in conf_backup.items():
            p.write_bytes(data)
        print(f"\nrestored canonical {SPLITS_PATH.name} and "
              f"{len(conf_backup)} confusion matrices", flush=True)


if __name__ == "__main__":
    main()
