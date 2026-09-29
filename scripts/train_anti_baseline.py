"""Anti-baseline: fit the Ridge probe on shuffled-y pairs, evaluate on real test.

Pipeline-leak sanity gate from the original framework note archived at
docs/archive/project-history/framework.md. If the scrambled fit still scores
non-trivially on real test data (R^2 clearly above zero, or cosine close to
the probe's), the real probe's numbers are suspect and something is joining Y
to X wrong upstream. Same cell as train_probe.py, with train+val targets
permuted (``run_cell(label_seed=...)``: one generator permutes train, then val).
Before Phase 1A each split had its own generator, so the val permutation, and
the recorded control, differ from the May runs for the same seed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from linear_trainer.cell import append_record, run_cell
from linear_trainer.protocol import V2
from train_probe import print_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PRED_ROOT = REPO_ROOT / "outputs" / "predictions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--splits", default=str(DATA / "splits.json"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    ap.add_argument("--pred-dir", default=None,
                    help="where test predictions go (default: outputs/predictions/<metrics stem>/)")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir) if args.pred_dir else PRED_ROOT / Path(args.metrics_out).stem

    dataset_path = Path(args.dataset)
    print(f"=== anti-baseline: {dataset_path.name} (Y shuffled in train + val) ===")
    res = run_cell(dataset_path, "genept", Path(args.splits), V2, pred_dir=pred_dir,
                   label_seed=args.shuffle_seed)
    print_sweep(res, "r2")
    print("\n  sanity: R^2 should be near zero; cosine should be well below the probe's.\n"
          "  If not, pipeline is leaking.")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"anti_baseline_{ts}",
        "timestamp": ts,
        "model": "anti_baseline_shuffled_y",
        "dataset": dataset_path.name,
        "shuffle_seed": args.shuffle_seed,
        "alpha": res["hp"],
        "alpha_sweep": res["sweep"],
        **res["metrics"],
        **res["provenance"],
    }
    append_record(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
