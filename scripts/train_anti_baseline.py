"""Anti-baseline: fit the linear probe on shuffled-y pairs, evaluate on real test.

Pipeline-leak sanity gate from the original framework note archived at
docs/archive/project-history/framework.md. If the scrambled fit still scores
non-trivially on real test data (R^2 clearly above zero, or cosine close to
the probe's), the real probe's numbers are suspect and something is joining Y
to X wrong upstream.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from linear_trainer import fit, sweep_alpha
from splits import load_shuffled_y, load_split

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
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    print(f"=== loading splits from {dataset_path.name} (Y shuffled in train + val) ===")
    X_tr, Y_tr_shuf, _ = load_shuffled_y("train", seed=args.shuffle_seed, dataset_path=dataset_path)
    X_val, Y_val_shuf, _ = load_shuffled_y("val", seed=args.shuffle_seed, dataset_path=dataset_path)
    X_te, Y_te, _ = load_split("test", dataset_path=dataset_path)
    print(f"  train={X_tr.shape} val={X_val.shape} test={X_te.shape}")

    print("\n=== alpha sweep on shuffled data (mean cosine on shuffled val) ===")
    best_alpha, sweep = sweep_alpha(X_tr, Y_tr_shuf, X_val, Y_val_shuf, args.alphas)
    for r in sweep:
        mark = " *" if r["alpha"] == best_alpha else ""
        print(f"  alpha={r['alpha']:>8.3g}  mean_cosine={r['mean_cosine']:.4f}{mark}")
    print(f"  best alpha = {best_alpha}")

    print("\n=== refit on shuffled train+val ===")
    X_fit = np.vstack([X_tr, X_val])
    Y_fit = np.vstack([Y_tr_shuf, Y_val_shuf])
    probe = fit(X_fit, Y_fit, best_alpha)

    print("\n=== evaluate on REAL test Y ===")
    Y_hat = probe.predict(X_te)
    cos = _cosine(Y_hat, Y_te)
    test_mean_cos = float(cos.mean())
    test_median_cos = float(np.median(cos))
    test_r2_macro = float(r2_score(Y_te, Y_hat, multioutput="uniform_average"))
    print(f"  test_mean_cosine   = {test_mean_cos:.4f}")
    print(f"  test_median_cosine = {test_median_cos:.4f}")
    print(f"  test_r2_macro      = {test_r2_macro:.4f}")
    print(
        "\n  sanity: R^2 should be near zero; cosine should be well below the probe's.\n"
        "  If not, pipeline is leaking."
    )

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"anti_baseline_{ts}",
        "timestamp": ts,
        "model": "anti_baseline_shuffled_y",
        "dataset": dataset_path.name,
        "shuffle_seed": args.shuffle_seed,
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
