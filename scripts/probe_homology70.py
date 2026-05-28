"""Probe the headline cells on the stricter 70%-identity homology split (MINA #1).

The primary analysis uses a 40%-identity MMseqs2 cluster split (``data/splits.json``).
A reviewer asked for a stricter supplementary split; ``data/splits_homology70.json``
was built at 70% identity but never probed. This swaps it in (backing up and
restoring the canonical split in a ``finally`` block), re-probes the same headline
cells ``seed_sensitivity.py`` uses, and prints a 70%-vs-40% comparison.

Headline metrics are written to ``data/metrics_homology70.json`` — the canonical
``data/metrics_homology.json`` is NOT touched. The classification cells overwrite the
fixed ``data/confusion_5way_*.json`` paths (a known side effect, see
``train_logistic_probe.py``); restore them after the run with
``git checkout -- data/confusion_5way_*.json`` and ``rm`` any newly-created untracked
ones.

Run: uv run scripts/probe_homology70.py
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

from seed_sensitivity import _run_cells, _harvest, SPLITS  # noqa: E402

SPLIT70 = DATA / "splits_homology70.json"
OUT = DATA / "metrics_homology70.json"
PRIMARY = DATA / "metrics_homology.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=SPLIT70)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.split.exists():
        raise SystemExit(f"missing split file: {args.split}")

    backup = SPLITS.read_bytes() if SPLITS.exists() else None
    try:
        SPLITS.write_bytes(args.split.read_bytes())
        s = json.loads(SPLITS.read_text())
        print(f"=== swapped in {args.split.name}: train={len(s['train'])} "
              f"val={len(s['val'])} test={len(s['test'])} ===", flush=True)
        if args.out.exists():
            args.out.unlink()
        _run_cells(args.out)
    finally:
        if backup is not None:
            SPLITS.write_bytes(backup)
            print(f"\nrestored canonical {SPLITS.name}", flush=True)

    h70 = _harvest(args.out)
    h40 = _harvest(PRIMARY) if PRIMARY.exists() else {"cls": {}, "reg": {}}
    print("\n=== 70% vs 40% homology split (headline cells) ===")
    for task, metric in (("cls", "macro-F1"), ("reg", "R^2")):
        print(f"\n  [{task}] {metric}:        70%   |    40%")
        for cell, v70 in sorted(h70[task].items(), key=lambda kv: -kv[1]):
            v40 = h40[task].get(cell)
            v40s = f"{v40:+.4f}" if v40 is not None else "   —   "
            print(f"    {cell:<28s} {v70:+.4f}  |  {v40s}")
    print("\nNOTE: restore clobbered confusion matrices: "
          "git checkout -- data/confusion_5way_*.json")


if __name__ == "__main__":
    main()
