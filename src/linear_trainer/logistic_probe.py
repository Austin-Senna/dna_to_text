"""Fitted logistic-regression classification probe.

The fit itself lives in ``linear_trainer.fit`` (the single fit path). This is
the small fitted artefact callers pass around without pickling sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticProbe:
    W: np.ndarray         # (d_in, n_classes) for multinomial; (d_in, 1) for binary
    b: np.ndarray         # (n_classes,) for multinomial; (1,) for binary
    classes: np.ndarray   # (n_classes,) class labels in column order of W
    C: float
    mu: np.ndarray | None = None      # scaler mean; None when the protocol does not scale
    sigma: np.ndarray | None = None   # scaler scale
    n_iter: int | None = None
    converged: bool = True

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.W.dtype)
        return X if self.mu is None else (X - self.mu) / self.sigma

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self._z(X) @ self.W + self.b
        return scores[:, 0] if self.W.shape[1] == 1 else scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for X."""
        scores = self.decision_function(X)
        if self.W.shape[1] == 1:
            # binary case: sklearn stores a single coefficient row for class 1
            return np.where(scores >= 0, self.classes[1], self.classes[0])
        return self.classes[np.argmax(scores, axis=1)]
