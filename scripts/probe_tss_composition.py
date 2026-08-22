"""Probe the chunk-matched composition baselines (E5 camera-ready linchpin).

Runs the same family5 logistic protocol as the anchored re-probe on the composition
features (dataset_tss_<enc>_{chunk4mergc,chunk6mer}.parquet), both splits, so we can
compare: does k-mer/GC of the SAME anchored chunk match the encoder's embedding?
  - if a composition baseline ~= the encoder (esp. recovering kinase) -> E5 is composition
  - if it stays ~chance while gena_lm holds 0.43 -> gena_lm's signal is encoder-specific

Registers the datasets at runtime (no edit to train_logistic_probe). Homology ->
metrics_tss_composition.json; disjoint -> metrics_tss_composition_disjoint.json (safe
swap+restore of splits.json). Never writes tracked data/metrics.json.

Run: uv run scripts/probe_tss_composition.py
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
FEATURES = ["chunk4mergc", "chunk6mer"]
DISJOINT = DATA / "splits_tss_disjoint.json"
HOM_OUT = DATA / "metrics_tss_composition.json"
DISJ_OUT = DATA / "metrics_tss_composition_disjoint.json"


def _register() -> list[str]:
    names = []
    for enc in ENCODERS:
        for feat in FEATURES:
            p = DATA / f"dataset_tss_{enc}_{feat}.parquet"
            if p.exists():
                name = f"tss_{enc}_{feat}"
                tlp.DATASET_PATHS[name] = p
                names.append(name)
    return names


def _probe(name: str, metrics_out: Path) -> None:
    old = sys.argv
    sys.argv = ["train_logistic_probe.py", "--dataset", name, "--task", "family5",
                "--metrics-out", str(metrics_out)]
    try:
        tlp.main()
    finally:
        sys.argv = old


def _run_all(names: list[str], metrics_out: Path, label: str) -> None:
    print(f"\n===== {label}: {len(names)} composition cells -> {metrics_out.name} =====", flush=True)
    if metrics_out.exists():
        metrics_out.unlink()
    for name in names:
        print(f"\n########## {label} {name} [family5] ##########", flush=True)
        _probe(name, metrics_out)


def main() -> None:
    names = _register()
    if not names:
        raise SystemExit("no composition parquets found; run build_tss_composition_baseline.py first")

    _run_all(names, HOM_OUT, "HOMOLOGY")

    if not DISJOINT.exists():
        raise SystemExit(f"missing {DISJOINT}")
    split_backup = SPLITS_PATH.read_bytes()
    conf_backup = {p: p.read_bytes() for p in DATA.glob("confusion_5way_*.json")}
    try:
        SPLITS_PATH.write_bytes(DISJOINT.read_bytes())
        print(f"\n=== swapped in {DISJOINT.name} ===", flush=True)
        _run_all(names, DISJ_OUT, "DISJOINT")
    finally:
        SPLITS_PATH.write_bytes(split_backup)
        for p, data in conf_backup.items():
            p.write_bytes(data)
        print(f"\nrestored canonical {SPLITS_PATH.name} and {len(conf_backup)} confusion matrices", flush=True)


if __name__ == "__main__":
    main()
