"""Train the linear probe: sweep alpha on val, refit on train+val, evaluate on test."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from linear_trainer import LinearProbe, fit, sweep_alpha
from splits import load_split

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
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--probe-out", default=str(DATA / "probe.npz"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    print(f"=== loading splits from {dataset_path.name} ===")
    X_tr, Y_tr, _ = load_split("train", dataset_path=dataset_path)
    X_val, Y_val, _ = load_split("val", dataset_path=dataset_path)
    X_te, Y_te, _ = load_split("test", dataset_path=dataset_path)
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
    assert probe.W.shape == (X_tr.shape[1], Y_tr.shape[1]), probe.W.shape
    assert probe.b.shape == (Y_tr.shape[1],), probe.b.shape

    print("\n=== evaluate on test ===")
    Y_hat = probe.predict(X_te)
    cos = _cosine(Y_hat, Y_te)
    test_mean_cos = float(cos.mean())
    test_median_cos = float(np.median(cos))
    test_r2_macro = float(r2_score(Y_te, Y_hat, multioutput="uniform_average"))
    assert test_mean_cos > 0, f"pipeline broken: mean cosine {test_mean_cos}"
    print(f"  test_mean_cosine   = {test_mean_cos:.4f}")
    print(f"  test_median_cosine = {test_median_cos:.4f}")
    print(f"  test_r2_macro      = {test_r2_macro:.4f}")

    probe_path = Path(args.probe_out)
    probe.save(probe_path)
    print(f"\n  wrote probe → {probe_path}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"probe_{ts}",
        "timestamp": ts,
        "model": "linear_probe",
        "dataset": dataset_path.name,
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
