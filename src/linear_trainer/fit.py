"""The single fit path: every probe fit, sweep and refit goes through here.

Probes, bootstraps and downstream refits used to carry their own copies of the
recipe (max_iter 2000/3000/5000, scaled or not, their own grids). One path
means one protocol; tests/test_one_fit_path.py keeps it that way.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from linear_trainer.logistic_probe import LogisticProbe
from linear_trainer.probe import LinearProbe, _mean_cosine
from linear_trainer.protocol import Protocol

KINDS = ("logistic", "ridge")


class ConvergenceFailure(RuntimeError):
    """A fit that must converge (a refit, or every point of a sweep) did not."""


def fit(kind: str, X: np.ndarray, y: np.ndarray, hp: float, protocol: Protocol,
        *, strict: bool = True) -> LogisticProbe | LinearProbe:
    """Fit one probe at hyperparameter ``hp`` (C for logistic, alpha for ridge).

    The scaler is kept on the probe, not folded into W and b: folding cancels
    about 7 digits when |mean|/std reaches 1e6-1e7 (HyenaDNA clsmean does).
    """
    X = np.asarray(X, dtype=protocol.dtype)
    mu = sigma = None
    if protocol.scale:
        scaler = StandardScaler().fit(X)
        mu, sigma = scaler.mean_.astype(protocol.dtype), scaler.scale_.astype(protocol.dtype)
        X = scaler.transform(X)
    if kind == "logistic":
        kw = {} if protocol.tol is None else {"tol": protocol.tol}
        est = LogisticRegression(C=hp, max_iter=protocol.max_iter, solver="lbfgs", **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)   # recorded, not printed
            est.fit(X, y)
        n_iter = int(np.max(est.n_iter_))
        converged = (n_iter < protocol.max_iter
                     and not any(issubclass(w.category, ConvergenceWarning) for w in caught))
        probe = LogisticProbe(W=est.coef_.T.astype(protocol.dtype),
                              b=est.intercept_.astype(protocol.dtype),
                              classes=est.classes_.copy(), C=float(hp), mu=mu, sigma=sigma,
                              n_iter=n_iter, converged=converged)
    elif kind == "ridge":
        est = Ridge(alpha=hp).fit(X, y)
        probe = LinearProbe(W=est.coef_.T.astype(protocol.dtype),
                            b=np.asarray(est.intercept_).astype(protocol.dtype),
                            alpha=float(hp), mu=mu, sigma=sigma, n_iter=None, converged=True)
    else:
        raise ValueError(f"unknown probe kind: {kind!r}")
    if strict and not probe.converged:
        raise ConvergenceFailure(
            f"{kind} fit at {hp:g} stopped at max_iter={protocol.max_iter} without converging")
    return probe


def score(kind: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Test metrics for one cell: the one metric function shared by cells and bootstraps."""
    if kind == "logistic":
        per_class = {}
        for c in sorted(set(y_true.tolist())):
            mask = y_true == c
            per_class[str(c)] = float((pred[mask] == c).mean()) if mask.any() else None
        return {
            "test_macro_f1": float(f1_score(y_true, pred, average="macro")),
            "test_kappa": float(cohen_kappa_score(y_true, pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "test_accuracy": float(accuracy_score(y_true, pred)),
            "test_per_class_accuracy": per_class,
        }
    cos = _cosines(pred, y_true)
    return {
        "test_mean_cosine": float(cos.mean()),
        "test_median_cosine": float(np.median(cos)),
        "test_r2_macro": float(r2_score(y_true, pred, multioutput="uniform_average")),
    }


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-12, None)


def _decade(hp: float) -> int:
    return round(math.log10(hp))


def extend_sweep(evaluate: Callable[[float], tuple], base_grid: Sequence[float],
                 limits: tuple[float, float], plateau_eps: float
                 ) -> tuple[list[dict], float, bool | str]:
    """Sweep ``base_grid``, then extend past an edge pick one decade at a time.

    ``evaluate(hp)`` returns ``(score, converged)`` or ``(score, converged, extra)``.
    The pick is the best converged point; ties go to the point evaluated first
    (the base grid in ascending order, then extensions). While the pick sits on
    the evaluated grid's edge, the next decade beyond it is tried:

    - it scores below the pick: the pick is interior, ``edge=False``;
    - it wins by more than ``plateau_eps``: keep extending;
    - it ties or wins by no more than ``plateau_eps``: stop, ``edge="plateau"``
      (a tie keeps the earlier pick; a flat curve never reads as interior);
    - the next decade is past ``limits``: stop, ``edge="limit"``;
    - it doesn't converge: stop, ``edge="nonconverged"``.

    A pick with a non-converged immediate neighbour is also flagged
    ``"nonconverged"``: the curve on that side is unknown. Returns (points in
    ascending hp, pick, edge).
    """
    points: list[dict] = []

    def run(hp: float) -> dict:
        out = evaluate(hp)
        point = {"hp": float(hp), "score": float(out[0]), "converged": bool(out[1]),
                 **(out[2] if len(out) > 2 else {})}
        points.append(point)
        return point

    def best() -> dict:
        eligible = [p for p in points if p["converged"]]
        if not eligible:
            raise ConvergenceFailure("no grid point converged")
        return max(eligible, key=lambda p: p["score"])

    for hp in sorted(base_grid):
        run(hp)
    lo, hi = limits
    edge: bool | str = False
    while True:
        pick = best()
        ordered = sorted(points, key=lambda p: p["hp"])
        i = next(k for k, p in enumerate(ordered) if p is pick)
        if 0 < i < len(ordered) - 1:
            if not (ordered[i - 1]["converged"] and ordered[i + 1]["converged"]):
                edge = "nonconverged"
            break
        hps = [p["hp"] for p in ordered]
        step = 1 if pick["hp"] == hps[-1] else -1
        nxt = float(f"1e{_decade(pick['hp']) + step}")
        if not lo <= nxt <= hi:
            edge = "limit"
            break
        new = run(nxt)
        if not new["converged"]:
            edge = "nonconverged"
            break
        if new["score"] > pick["score"] + plateau_eps:
            continue
        if new["score"] >= pick["score"]:
            edge = "plateau"
        break
    return sorted(points, key=lambda p: p["hp"]), best()["hp"], edge


@dataclass
class Selection:
    hp: float
    sweep: list[dict]        # legacy record shape: C_sweep / alpha_sweep rows, ascending
    grid: list[float]
    edge: bool | str


def select(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
           y_val: np.ndarray, protocol: Protocol, *, select_by: str = "r2") -> Selection:
    """Pick C (logistic, val macro-F1) or alpha (ridge, val R^2 or cosine) on validation.

    There is deliberately no test argument: selection never sees test data.
    """
    if kind == "ridge" and select_by not in ("r2", "cosine"):
        raise ValueError(f"select_by must be 'r2' or 'cosine', got {select_by!r}")

    def evaluate(hp: float):
        probe = fit(kind, X_tr, y_tr, hp, protocol, strict=False)
        y_hat = probe.predict(X_val)
        if kind == "logistic":
            row = {"C": float(hp), "macro_f1": float(f1_score(y_val, y_hat, average="macro"))}
            s = row["macro_f1"]
        else:
            row = {"alpha": float(hp), "mean_cosine": _mean_cosine(y_hat, y_val),
                   "r2": float(r2_score(y_val, y_hat, multioutput="uniform_average"))}
            s = row["r2"] if select_by == "r2" else row["mean_cosine"]
        row.update(n_iter=probe.n_iter, converged=probe.converged)
        return s, probe.converged, row

    hp_key = "C" if kind == "logistic" else "alpha"
    points, pick, edge = extend_sweep(evaluate, protocol.base_grid, protocol.limits[hp_key],
                                      protocol.plateau_eps)
    sweep = [{k: v for k, v in p.items() if k not in ("hp", "score")} for p in points]
    return Selection(hp=pick, sweep=sweep, grid=[p["hp"] for p in points], edge=edge)
