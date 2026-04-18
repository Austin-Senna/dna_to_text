"""Diagnostic MLP probe: same splits/eval as train_probe.py, nonlinear readout.

Answers: does a 1-hidden-layer MLP on DNABERT-2 embeddings extract signal a
linear probe cannot? If R^2 meaningfully clears the linear probe's number,
DNABERT-2 has nonlinear structure worth chasing with a bigger model or
fine-tune. If it lands at ~0.18, the task is the ceiling, not the model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from linear_trainer import fit_mlp, sweep_mlp
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
DEFAULT_HIDDEN = [(256,), (512,), (1024,)]
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2]


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
    ap.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        action="append",
        help="Hidden sizes to sweep; repeat flag for multiple (e.g. --hidden 256 --hidden 512).",
    )
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    hidden_configs = [tuple(h) for h in args.hidden] if args.hidden else DEFAULT_HIDDEN
    configs = [{"hidden": h, "alpha": a} for h, a in product(hidden_configs, args.alphas)]

    dataset_path = Path(args.dataset)
    print(f"=== loading splits from {dataset_path.name} ===")
    X_tr, Y_tr, _ = load_split("train", dataset_path=dataset_path)
    X_val, Y_val, _ = load_split("val", dataset_path=dataset_path)
    X_te, Y_te, _ = load_split("test", dataset_path=dataset_path)
    print(f"  train={X_tr.shape} val={X_val.shape} test={X_te.shape}")

    print(f"\n=== MLP sweep (mean cosine on val) — {len(configs)} configs ===")
    best_cfg, sweep_results = sweep_mlp(X_tr, Y_tr, X_val, Y_val, configs)
    for r in sweep_results:
        mark = (
            " *"
            if (tuple(r["hidden"]) == tuple(best_cfg["hidden"]) and r["alpha"] == best_cfg["alpha"])
            else ""
        )
        print(
            f"  hidden={str(tuple(r['hidden'])):>12} alpha={r['alpha']:>7.1e}"
            f"  mean_cosine={r['mean_cosine']:.4f}{mark}"
        )
    print(f"  best = hidden={tuple(best_cfg['hidden'])} alpha={best_cfg['alpha']}")

    print("\n=== refit on train+val ===")
    X_fit = np.vstack([X_tr, X_val])
    Y_fit = np.vstack([Y_tr, Y_val])
    probe = fit_mlp(X_fit, Y_fit, hidden=tuple(best_cfg["hidden"]), alpha=float(best_cfg["alpha"]))

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

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "run_id": f"mlp_probe_{ts}",
        "timestamp": ts,
        "model": "mlp_probe",
        "dataset": dataset_path.name,
        "hidden": list(best_cfg["hidden"]),
        "alpha": best_cfg["alpha"],
        "sweep": sweep_results,
        "test_mean_cosine": test_mean_cos,
        "test_median_cosine": test_median_cos,
        "test_r2_macro": test_r2_macro,
    }
    _append_metrics(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
