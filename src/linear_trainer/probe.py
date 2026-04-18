"""Ridge linear probe: W: R^d_in -> R^d_out.

Training is a pair of free functions; the fitted artefact is a small dataclass
so callers can pass it around (zero-shot, viz, interp) without pickling an
sklearn estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class LinearProbe:
    W: np.ndarray        # (d_in, d_out)
    b: np.ndarray        # (d_out,)
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, alpha=np.float64(self.alpha))

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        data = np.load(path)
        return cls(W=data["W"], b=data["b"], alpha=float(data["alpha"]))


def _mean_cosine(y_hat: np.ndarray, y: np.ndarray) -> float:
    num = (y_hat * y).sum(axis=-1)
    den = np.linalg.norm(y_hat, axis=-1) * np.linalg.norm(y, axis=-1)
    return float((num / np.clip(den, 1e-12, None)).mean())


def fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> LinearProbe:
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    # sklearn: coef_ is (d_out, d_in); we want W: (d_in, d_out) so X @ W + b works.
    W = model.coef_.T.astype(np.float32)
    b = model.intercept_.astype(np.float32)
    return LinearProbe(W=W, b=b, alpha=float(alpha))


def sweep_alpha(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, list[dict]]:
    """Fit at each alpha on train, score mean cosine on val. Return (best_alpha, results)."""
    results: list[dict] = []
    for a in alphas:
        probe = fit(X_tr, Y_tr, a)
        cos = _mean_cosine(probe.predict(X_val), Y_val)
        results.append({"alpha": float(a), "mean_cosine": cos})
    best = max(results, key=lambda r: r["mean_cosine"])
    return best["alpha"], results
