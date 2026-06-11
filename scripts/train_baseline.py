"""Train a compositional Ridge baseline (encoder embeddings -> GenePT vectors).

Same Ridge + alpha sweep recipe as train_probe.py but with a simple
compositional feature source instead of DNA-LM embeddings. ``--feature``
picks the source: the original CDS 4-mer plus the journal-revision baselines
(6-mer, codon frequency, translated amino-acid k-mer, GC/length).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from kmer_baseline import load_kmer_features
from composition_baseline import (
    load_aa_kmer_features,
    load_codon_features,
    load_gc_features,
)
from linear_trainer import fit, sweep_alpha

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
DEFAULT_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

# feature name -> (loader(name) -> (X, Y, meta), model label for metrics)
FEATURE_LOADERS = {
    "kmer": (lambda n: load_kmer_features(n, k=4), "kmer_baseline_4"),
    "kmer6": (lambda n: load_kmer_features(n, k=6), "kmer_baseline_6"),
    "codon": (load_codon_features, "codon_baseline"),
    "aa1": (lambda n: load_aa_kmer_features(n, k=1), "aa_baseline_1"),
    "aa2": (lambda n: load_aa_kmer_features(n, k=2), "aa_baseline_2"),
    "aa3": (lambda n: load_aa_kmer_features(n, k=3), "aa_baseline_3"),
    "gc": (load_gc_features, "gc_baseline"),
}


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-12, None)


def _append_metrics(path: Path, entry: dict) -> None:
    runs: list = []
    if path.exists():
        runs = json.loads(path.read_text())
        if not isinstance(runs, list):
            raise ValueError(f"{path} is not a JSON array")
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=sorted(FEATURE_LOADERS), default="kmer",
                    help="compositional feature source (default: CDS 4-mer)")
    ap.add_argument("--select-by", choices=["r2", "cosine"], default="r2",
                    help="validation metric for alpha selection (default: macro-R^2)")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    loader, model_label = FEATURE_LOADERS[args.feature]
    print(f"=== loading {args.feature} features ===")
    X_tr, Y_tr, _ = loader("train")
    X_val, Y_val, _ = loader("val")
    X_te, Y_te, _ = loader("test")
    print(f"  train={X_tr.shape} val={X_val.shape} test={X_te.shape}")

    print(f"\n=== alpha sweep (select by val {args.select_by}) ===")
    best_alpha, sweep = sweep_alpha(
        X_tr, Y_tr, X_val, Y_val, args.alphas, select_by=args.select_by
    )
    for r in sweep:
        mark = " *" if r["alpha"] == best_alpha else ""
        print(
            f"  alpha={r['alpha']:>8.3g}  "
            f"val_r2={r['r2']:.4f}  mean_cosine={r['mean_cosine']:.4f}{mark}"
        )
    print(f"  best alpha = {best_alpha}  (selected by val {args.select_by})")

    print("\n=== refit on train+val ===")
    X_fit = np.vstack([X_tr, X_val])
    Y_fit = np.vstack([Y_tr, Y_val])
    probe = fit(X_fit, Y_fit, best_alpha)
    assert probe.W.shape == (X_tr.shape[1], Y_tr.shape[1])

    print("\n=== evaluate on test ===")
    Y_hat = probe.predict(X_te)
    cos = _cosine(Y_hat, Y_te)
    test_mean_cos = float(cos.mean())
    test_median_cos = float(np.median(cos))
    test_r2_macro = float(r2_score(Y_te, Y_hat, multioutput="uniform_average"))
    print(f"  test_mean_cosine   = {test_mean_cos:.4f}")
    print(f"  test_median_cosine = {test_median_cos:.4f}")
    print(f"  test_r2_macro      = {test_r2_macro:.4f}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"{model_label}_{ts}",
        "timestamp": ts,
        "model": model_label,
        "feature_source": args.feature,
        "feature_dim": int(X_tr.shape[1]),
        "select_by": args.select_by,
        "alpha": best_alpha,
        "alpha_sweep": sweep,
        "test_mean_cosine": test_mean_cos,
        "test_median_cosine": test_median_cos,
        "test_r2_macro": test_r2_macro,
    }
    _append_metrics(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
