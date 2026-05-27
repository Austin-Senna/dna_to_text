"""Split-seed sensitivity over several homology-aware splits (MINA #10).

Clustering is seed-independent — only the whole-cluster -> train/val/test assignment
uses the seed. So we cluster ONCE (MMseqs2 @ primary id) and re-assign per seed, then
re-probe the headline cells that carry the paper's three conclusions:
  * CDS arm: AA-composition vs best DNA-LM (does composition keep up?)
  * TSS arm: does it stay at the floor (collapse) across seeds?
  * ESM-2: does it keep winning?

The canonical ``data/splits.json`` is backed up and restored in a finally block (it is
also git-tracked, so recoverable regardless). Per-seed metrics land in
``data/seed_sensitivity/metrics_seed{N}.json``; a summary is printed and written to
``data/seed_sensitivity/summary.json``.

Run: uv run scripts/seed_sensitivity.py --seeds 1 7 123
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

import pandas as pd  # noqa: E402

import train_logistic_probe as tlp  # noqa: E402
import train_baseline as tb  # noqa: E402
import train_probe as tp  # noqa: E402
from cluster import cluster_dataframe  # noqa: E402
from splits.loader import resolve_dataset_path  # noqa: E402
from splits.make_splits import write_cluster_splits_json  # noqa: E402

SPLITS = DATA / "splits.json"

# Headline cells (carry the three conclusions); kept small so N seeds stays cheap.
CLS_CELLS = ["nt_v2_meanG", "aa2", "kmer", "tss_dnabert2_meanmean", "esm2_150m", "esm2_650m"]
REG_PARQUETS = ["dataset_dnabert2_meanD.parquet", "dataset_tss_dnabert2_meanmean.parquet",
                "dataset_esm2_150m.parquet", "dataset_esm2_650m.parquet"]
REG_BASELINES = ["aa3"]


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
        if ds in tlp.SYNTHETIC_FEATURIZERS or (tlp.DATASET_PATHS.get(ds) and Path(tlp.DATASET_PATHS[ds]).exists()):
            _call(tlp, ["train_logistic_probe.py", "--dataset", ds, "--task", "family5",
                        "--metrics-out", str(out)])
    for feat in REG_BASELINES:
        _call(tb, ["train_baseline.py", "--feature", feat, "--metrics-out", str(out)])
    for fn in REG_PARQUETS:
        if (DATA / fn).exists():
            _call(tp, ["train_probe.py", "--dataset", str(DATA / fn),
                       "--probe-out", "/tmp/_seedsens_probe.npz", "--metrics-out", str(out)])


def _harvest(metrics_path: Path) -> dict:
    """Latest macro-F1 (cls) / R² (reg) per cell from a metrics file."""
    M = json.loads(metrics_path.read_text())
    cls, reg = {}, {}
    for e in M:
        if e.get("model") == "logistic_probe" and e.get("task") == "family5":
            cls[e["feature_source"]] = e["test_macro_f1"]
        elif e.get("model") == "linear_probe":
            reg[Path(e["dataset"]).stem.replace("dataset_", "")] = e["test_r2_macro"]
        elif "baseline" in str(e.get("model", "")):
            reg[e["feature_source"]] = e["test_r2_macro"]
    return {"cls": cls, "reg": reg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 123])
    ap.add_argument("--primary-id", type=float, default=0.40)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--workdir", default=str(DATA / "cluster_work"))
    args = ap.parse_args()

    out_dir = DATA / "seed_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(resolve_dataset_path(), columns=["ensembl_id", "family"]).drop_duplicates("ensembl_id")
    print(f"=== clustering once @ id>={args.primary_id} over {len(df)} genes ===", flush=True)
    df_clustered, stats = cluster_dataframe(
        df, Path(args.workdir) / f"id{int(args.primary_id * 100)}",
        min_seq_id=args.primary_id, coverage=args.coverage,
    )

    backup = SPLITS.read_bytes() if SPLITS.exists() else None
    summary: dict[str, dict] = {}
    try:
        for seed in args.seeds:
            print(f"\n########## SEED {seed} ##########", flush=True)
            payload = write_cluster_splits_json(df_clustered, SPLITS, stats, seed=seed)
            print(f"  split sizes: train={len(payload['train'])} val={len(payload['val'])} "
                  f"test={len(payload['test'])}", flush=True)
            metrics_path = out_dir / f"metrics_seed{seed}.json"
            if metrics_path.exists():
                metrics_path.unlink()
            _run_cells(metrics_path)
            summary[str(seed)] = _harvest(metrics_path)
            print(f"  seed {seed}: {summary[str(seed)]}", flush=True)
    finally:
        if backup is not None:
            SPLITS.write_bytes(backup)
            print(f"\nrestored canonical {SPLITS.name}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
