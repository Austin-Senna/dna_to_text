"""Train the Ridge probe (encoder embeddings -> GenePT vectors) for one parquet.

Sweep alpha on val, refit on train+val, score test once: ``run_cell`` under
the V2 protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from linear_trainer.cell import append_record, run_cell
from linear_trainer.protocol import V2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PRED_ROOT = REPO_ROOT / "outputs" / "predictions"


def print_sweep(res: dict, select_by: str) -> None:
    for r in res["sweep"]:
        mark = " *" if r["alpha"] == res["hp"] else ""
        print(f"  alpha={r['alpha']:>8.3g}  val_r2={r['r2']:.4f}  "
              f"mean_cosine={r['mean_cosine']:.4f}{mark}")
    m = res["metrics"]
    print(f"  best alpha = {res['hp']:g}  (selected by val {select_by}; edge={res['edge']})")
    print(f"  test_mean_cosine = {m['test_mean_cosine']:.4f}  "
          f"test_median_cosine = {m['test_median_cosine']:.4f}  test_r2_macro = {m['test_r2_macro']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--select-by",
        choices=["r2", "cosine"],
        default="r2",
        help="validation metric for alpha selection (default: macro-R^2)",
    )
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--probe-out", default=str(DATA / "probe.npz"))
    ap.add_argument("--splits", default=str(DATA / "splits.json"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    ap.add_argument("--pred-dir", default=None,
                    help="where test predictions go (default: outputs/predictions/<metrics stem>/)")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir) if args.pred_dir else PRED_ROOT / Path(args.metrics_out).stem

    dataset_path = Path(args.dataset)
    print(f"=== Ridge cell: {dataset_path.name} splits={args.splits} ===")
    res = run_cell(dataset_path, "genept", Path(args.splits), V2, pred_dir=pred_dir,
                   select_by=args.select_by, probe_out=Path(args.probe_out))
    print_sweep(res, args.select_by)
    m = res["metrics"]
    assert m["test_mean_cosine"] > 0, f"pipeline broken: mean cosine {m['test_mean_cosine']}"

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"probe_{ts}",
        "timestamp": ts,
        "model": "linear_probe",
        "dataset": dataset_path.name,
        "alpha": res["hp"],
        "select_by": args.select_by,
        "alpha_sweep": res["sweep"],
        **m,
        **res["provenance"],
    }
    append_record(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
