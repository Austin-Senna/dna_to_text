"""Logistic regression classification probe with L2 + C-sweep.

Mirrors the discipline of probe.py (Ridge): sweep C on val, refit on
train+val, evaluate on test. Multi-class via multinomial softmax;
binary via the same path (sklearn handles the dispatch internally).

Headline metric for the C-sweep is macro-F1 (not accuracy) so the sweep
optimises for the same metric we report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


@dataclass
class LogisticProbe:
    W: np.ndarray         # (d_in, n_classes) for multinomial; (d_in, 1) for binary
    b: np.ndarray         # (n_classes,) for multinomial; (1,) for binary
    classes: np.ndarray   # (n_classes,) class labels in column order of W
    C: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for X."""
        if self.W.shape[1] == 1:
            # binary case: sklearn stores a single coefficient row for class 1
            logit = X @ self.W[:, 0] + self.b[0]
            return np.where(logit >= 0, self.classes[1], self.classes[0])
        scores = X @ self.W + self.b  # (n, n_classes)
        return self.classes[np.argmax(scores, axis=1)]


def _make_logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs")


def fit(X: np.ndarray, y: np.ndarray, C: float) -> LogisticProbe:
    model = _make_logreg(C)
    model.fit(X, y)
    W = model.coef_.T.astype(np.float32)               # (d_in, n_classes_or_1)
    b = model.intercept_.astype(np.float32)            # (n_classes_or_1,)
    return LogisticProbe(W=W, b=b, classes=model.classes_.copy(), C=float(C))


def sweep_C(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Cs: Sequence[float],
) -> tuple[float, list[dict]]:
    """Fit at each C on train, score macro-F1 on val. Return (best_C, results)."""
    results: list[dict] = []
    for C in Cs:
        probe = fit(X_tr, y_tr, C)
        y_hat = probe.predict(X_val)
        macro_f1 = float(f1_score(y_val, y_hat, average="macro"))
        results.append({"C": float(C), "macro_f1": macro_f1})
    best = max(results, key=lambda r: r["macro_f1"])
    return best["C"], results
