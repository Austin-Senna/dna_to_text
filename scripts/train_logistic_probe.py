"""Train a logistic regression classification probe (one cell).

Flags pick which feature source X comes from (--dataset) and which
classification task (--task); --shuffle-labels turns the run into the
anti-baseline. The fit, validation sweep and test scoring are
``linear_trainer.cell.run_cell`` under the V2 protocol.

Outputs:
  - one record appended to --metrics-out per run
  - the cell's test predictions under --pred-dir (bootstraps rescore these)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from linear_trainer import sources
from linear_trainer.cell import append_record, run_cell
from linear_trainer.protocol import V2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PRED_ROOT = REPO_ROOT / "outputs" / "predictions"

# The registry lives in linear_trainer.sources; these are the same objects, so a
# script that registers a parquet here is seen by every reader.
DATASET_PATHS = sources.DATASET_PATHS
SYNTHETIC_FEATURIZERS = sources.SYNTHETIC_FEATURIZERS
META_PARQUET = sources.META_PARQUET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=sorted(set(DATASET_PATHS.keys()) | set(SYNTHETIC_FEATURIZERS)))
    ap.add_argument("--task", required=True,
                    choices=["family5", "tf_vs_gpcr", "tf_vs_kinase"])
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="anti-baseline: permute y in train+val before fit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default=str(DATA / "splits.json"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    ap.add_argument("--pred-dir", default=None,
                    help="where test predictions go (default: outputs/predictions/<metrics stem>/)")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir) if args.pred_dir else PRED_ROOT / Path(args.metrics_out).stem

    print(f"=== cell: dataset={args.dataset} task={args.task} splits={args.splits} ===")
    res = run_cell(args.dataset, args.task, Path(args.splits), V2, pred_dir=pred_dir,
                   label_seed=args.seed if args.shuffle_labels else None)
    for r in res["sweep"]:
        mark = " *" if r["C"] == res["hp"] else ""
        flag = "" if r["converged"] else "  (not converged)"
        print(f"  C={r['C']:>8.3g}  val macro_f1={r['macro_f1']:.4f}{mark}{flag}")
    m = res["metrics"]
    print(f"  best C = {res['hp']:g}  edge={res['edge']}")
    print(f"  test_macro_f1 = {m['test_macro_f1']:.4f}  kappa = {m['test_kappa']:.4f}  "
          f"balanced_acc = {m['test_balanced_accuracy']:.4f}  acc = {m['test_accuracy']:.4f}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    encoder_label = "shuffled" if args.shuffle_labels else args.dataset
    entry = {
        "run_id": f"logistic_{encoder_label}_{args.task}_{ts}",
        "timestamp": ts,
        "model": "logistic_probe",
        "encoder": encoder_label,
        "feature_source": args.dataset,
        "task": args.task,
        "shuffled_labels": args.shuffle_labels,
        "C": res["hp"],
        "C_sweep": res["sweep"],
        **m,
        **res["provenance"],
    }
    append_record(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
