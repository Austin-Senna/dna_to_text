"""Fitted Ridge linear probe: W: R^d_in -> R^d_out.

The fit itself lives in ``linear_trainer.fit`` (the single fit path). The
fitted artefact is a small dataclass so callers can pass it around (zero-shot,
viz, interp) without pickling an sklearn estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearProbe:
    W: np.ndarray        # (d_in, d_out)
    b: np.ndarray        # (d_out,)
    alpha: float
    mu: np.ndarray | None = None      # scaler mean; None when the protocol does not scale
    sigma: np.ndarray | None = None   # scaler scale
    n_iter: int | None = None
    converged: bool = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.W.dtype)
        if self.mu is not None:
            X = (X - self.mu) / self.sigma
        return X @ self.W + self.b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.mu is None else {"mu": self.mu, "sigma": self.sigma}
        np.savez(path, W=self.W, b=self.b, alpha=np.float64(self.alpha), **extra)

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        data = np.load(path)
        mu = data["mu"] if "mu" in data else None
        sigma = data["sigma"] if "sigma" in data else None
        return cls(W=data["W"], b=data["b"], alpha=float(data["alpha"]), mu=mu, sigma=sigma)


def _mean_cosine(y_hat: np.ndarray, y: np.ndarray) -> float:
    num = (y_hat * y).sum(axis=-1)
    den = np.linalg.norm(y_hat, axis=-1) * np.linalg.norm(y, axis=-1)
    return float((num / np.clip(den, 1e-12, None)).mean())
