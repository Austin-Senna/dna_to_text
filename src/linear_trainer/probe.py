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
from sklearn.metrics import r2_score


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
    select_by: str = "r2",
) -> tuple[float, list[dict]]:
    """Fit at each alpha on train, score on val. Return (best_alpha, results).

    Selection metric is validation macro-R^2 by default (``select_by="r2"``),
    matching the manuscript's argument that R^2 — not the compressed cosine
    similarity — is the interpreted regression metric. Mean cosine is still
    recorded per alpha as a secondary diagnostic. Pass ``select_by="cosine"``
    to recover the legacy cosine-selected behaviour (used for the
    α-selection sensitivity analysis).
    """
    if select_by not in ("r2", "cosine"):
        raise ValueError(f"select_by must be 'r2' or 'cosine', got {select_by!r}")
    results: list[dict] = []
    for a in alphas:
        probe = fit(X_tr, Y_tr, a)
        Y_hat = probe.predict(X_val)
        cos = _mean_cosine(Y_hat, Y_val)
        r2 = float(r2_score(Y_val, Y_hat, multioutput="uniform_average"))
        results.append({"alpha": float(a), "mean_cosine": cos, "r2": r2})
    key = "r2" if select_by == "r2" else "mean_cosine"
    best = max(results, key=lambda r: r[key])
    return best["alpha"], results
