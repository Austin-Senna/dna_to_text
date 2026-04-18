"""Train the 4-mer k-mer baseline. Same Ridge + alpha sweep recipe as train_probe.py
but with 256-d L1-normalised 4-mer frequency features instead of DNABERT-2 embeddings.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from kmer_baseline import KMER_DIM, load_kmer_features
from linear_trainer import fit, sweep_alpha

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
DEFAULT_ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


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
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    print("=== loading 4-mer features ===")
    X_tr, Y_tr, _ = load_kmer_features("train")
    X_val, Y_val, _ = load_kmer_features("val")
    X_te, Y_te, _ = load_kmer_features("test")
    print(f"  train={X_tr.shape} val={X_val.shape} test={X_te.shape}")

    print("\n=== alpha sweep (mean cosine on val) ===")
    best_alpha, sweep = sweep_alpha(X_tr, Y_tr, X_val, Y_val, args.alphas)
    for r in sweep:
        mark = " *" if r["alpha"] == best_alpha else ""
        print(f"  alpha={r['alpha']:>8.3g}  mean_cosine={r['mean_cosine']:.4f}{mark}")
    print(f"  best alpha = {best_alpha}")

    print("\n=== refit on train+val ===")
    X_fit = np.vstack([X_tr, X_val])
    Y_fit = np.vstack([Y_tr, Y_val])
    probe = fit(X_fit, Y_fit, best_alpha)
    assert probe.W.shape == (KMER_DIM, Y_tr.shape[1])

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
        "run_id": f"kmer_baseline_{ts}",
        "timestamp": ts,
        "model": "kmer_baseline_4",
        "feature_dim": KMER_DIM,
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
