"""MLP probe: diagnostic for nonlinear signal in DNABERT-2 embeddings.

Parallel structure to probe.py. Uses sklearn's MLPRegressor so this stays a
one-file diagnostic — no torch training loop, no save/load plumbing. If the
MLP clears the linear probe's R^2 by a meaningful margin, it's worth building
a proper PyTorch version with GPU + persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor

from linear_trainer.probe import _mean_cosine


@dataclass
class MLPProbe:
    model: MLPRegressor
    hidden: tuple[int, ...]
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def fit(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: tuple[int, ...],
    alpha: float,
    max_iter: int = 1000,
    random_state: int = 0,
) -> MLPProbe:
    # Why no StandardScaler on X: DNABERT-2 embedding dims aren't independent
    # Gaussians; per-dim z-scoring strips meaningful signed means and equalises
    # informative variances, and empirically dropped val cosine from 0.93 to
    # 0.90 while stretching convergence from 62 to 980 iters.
    #
    # early_stopping peels off 10% of X as an internal val set for the MLP's own
    # convergence check; outer sweep still picks configs on the caller's X_val.
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )
    model.fit(X, Y)
    return MLPProbe(model=model, hidden=tuple(hidden), alpha=float(alpha))


def sweep(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    configs: Sequence[dict],
) -> tuple[dict, list[dict]]:
    """Fit each config on train, score mean cosine on val. Return (best_config, results)."""
    results: list[dict] = []
    for cfg in configs:
        probe = fit(X_tr, Y_tr, hidden=tuple(cfg["hidden"]), alpha=float(cfg["alpha"]))
        cos = _mean_cosine(probe.predict(X_val), Y_val)
        results.append(
            {"hidden": list(cfg["hidden"]), "alpha": float(cfg["alpha"]), "mean_cosine": cos}
        )
    best = max(results, key=lambda r: r["mean_cosine"])
    return {"hidden": best["hidden"], "alpha": best["alpha"]}, results
