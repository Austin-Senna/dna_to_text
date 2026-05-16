"""Post-extraction runner: build pooling datasets and train probes for one TSS encoder.

After `run_tss_multi_pool_extract.py --encoder <enc>` finishes, this script:
  1. Materialises pooling-variant parquets via `build_tss_pooling_datasets.py`.
  2. Trains the 5-way family-classification logistic probe for each variant.
  3. Trains the GenePT-regression ridge probe for each variant.
  4. Reports the best pool per task by reading the appended metrics.json entries.

Idempotent: probes that already appear in metrics.json with the same dataset
and task are skipped (the underlying train_*.py scripts do not currently dedup,
so the responsibility is here).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PY = sys.executable  # uv invocation already established the venv interp


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {' '.join(cmd)}")


def list_built_variants(encoder: str) -> list[str]:
    prefix = f"dataset_tss_{encoder}_"
    return sorted(
        p.stem.removeprefix(prefix)
        for p in DATA.glob(f"{prefix}*.parquet")
    )


def load_metrics() -> list[dict]:
    p = DATA / "metrics.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def already_done(metrics: list[dict], dataset_name: str, task: str | None) -> bool:
    for r in metrics:
        ds = r.get("dataset", "")
        if dataset_name in ds:
            if task is None and r.get("model") == "linear_probe":
                return True
            if task and r.get("task") == task:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="one of dnabert2 / nt_v2 / gena_lm / hyena_dna")
    ap.add_argument("--skip-build", action="store_true",
                    help="assume pooling parquets already built")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip probes whose metric entry already exists")
    args = ap.parse_args()

    enc = args.encoder

    if not args.skip_build:
        run([PY, "scripts/build_tss_pooling_datasets.py", "--encoder", enc])

    variants = list_built_variants(enc)
    if not variants:
        raise SystemExit(f"no dataset_tss_{enc}_*.parquet built; run extraction first")

    # Exclude the bare alias (no variant suffix); only keep the suffixed ones
    variants = [v for v in variants if v]
    print(f"\n=== Built pooling variants for {enc}: {variants} ===")

    metrics_before = load_metrics()

    print(f"\n=== Logistic probes (family5) for {len(variants)} variants ===")
    for v in variants:
        dataset_name = f"tss_{enc}_{v}"
        if args.skip_existing and already_done(metrics_before, dataset_name, task="family5"):
            print(f"  skip {dataset_name} family5 (already in metrics.json)")
            continue
        run([PY, "scripts/train_logistic_probe.py",
             "--dataset", dataset_name, "--task", "family5"])

    print(f"\n=== Ridge probes (genept regression) for {len(variants)} variants ===")
    for v in variants:
        parquet = DATA / f"dataset_tss_{enc}_{v}.parquet"
        if args.skip_existing and already_done(metrics_before, parquet.name, task=None):
            print(f"  skip {parquet.name} ridge (already in metrics.json)")
            continue
        run([PY, "scripts/train_probe.py",
             "--dataset", str(parquet),
             "--probe-out", str(DATA / f"probe_tss_{enc}_{v}.npz")])

    print(f"\n=== Best pool per task for {enc} ===")
    metrics = load_metrics()
    # logistic_probe records use 'feature_source'/'encoder' fields; ridge uses 'dataset'
    cls_rows = [r for r in metrics
                if r.get("model") == "logistic_probe"
                and f"tss_{enc}_" in (r.get("feature_source") or r.get("encoder") or "")]
    reg_rows = [r for r in metrics
                if r.get("model") == "linear_probe"
                and f"tss_{enc}_" in r.get("dataset", "")]

    if cls_rows:
        best_cls = max(cls_rows, key=lambda r: r.get("test_macro_f1", -1))
        label = best_cls.get("feature_source") or best_cls.get("encoder", "?")
        print(f"  family5 best: {label:40s} "
              f"F1={best_cls.get('test_macro_f1','?')}  "
              f"kappa={best_cls.get('test_kappa','?')}  "
              f"C={best_cls.get('C', '?')}")
    if reg_rows:
        best_reg = max(reg_rows, key=lambda r: r.get("test_r2_macro", -1))
        print(f"  genept  best: {best_reg.get('dataset','?'):40s} "
              f"R2={best_reg.get('test_r2_macro','?')}  "
              f"alpha={best_reg.get('alpha','?')}")


if __name__ == "__main__":
    main()
