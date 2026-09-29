"""Train a compositional Ridge baseline (composition features -> GenePT vectors).

Same cell as train_probe.py (``run_cell``, V2 protocol) with a compositional
feature source instead of DNA-LM embeddings. ``--feature`` picks the source:
the original CDS 4-mer plus the journal-revision baselines (6-mer, codon
frequency, translated amino-acid k-mer, GC/length).
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

# feature name -> model label for metrics
MODEL_LABELS = {
    "kmer": "kmer_baseline_4",
    "kmer6": "kmer_baseline_6",
    "codon": "codon_baseline",
    "aa1": "aa_baseline_1",
    "aa2": "aa_baseline_2",
    "aa3": "aa_baseline_3",
    "gc": "gc_baseline",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=sorted(MODEL_LABELS), default="kmer",
                    help="compositional feature source (default: CDS 4-mer)")
    ap.add_argument("--select-by", choices=["r2", "cosine"], default="r2",
                    help="validation metric for alpha selection (default: macro-R^2)")
    ap.add_argument("--splits", default=str(DATA / "splits.json"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    ap.add_argument("--pred-dir", default=None,
                    help="where test predictions go (default: outputs/predictions/<metrics stem>/)")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir) if args.pred_dir else PRED_ROOT / Path(args.metrics_out).stem

    print(f"=== Ridge baseline cell: {args.feature} splits={args.splits} ===")
    res = run_cell(args.feature, "genept", Path(args.splits), V2, pred_dir=pred_dir,
                   select_by=args.select_by)
    print_sweep(res, args.select_by)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_label = MODEL_LABELS[args.feature]
    entry = {
        "run_id": f"{model_label}_{ts}",
        "timestamp": ts,
        "model": model_label,
        "feature_source": args.feature,
        "feature_dim": res["feature_dim"],
        "select_by": args.select_by,
        "alpha": res["hp"],
        "alpha_sweep": res["sweep"],
        **res["metrics"],
        **res["provenance"],
    }
    append_record(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
