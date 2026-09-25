"""Validation-based model selection over recorded probe runs.

Every probe run records its hyperparameter sweep on the validation split
(``C_sweep`` for logistic probes, ``alpha_sweep`` for Ridge). Choosing a pooling
rule (or any other configuration) for an encoder must use those validation
scores, never the test metrics, so the held-out test split is scored exactly
once per reported cell.
"""
from __future__ import annotations

from typing import Iterable


def val_score(rec: dict) -> float:
    """Best validation score in a run's sweep (the value its hyperparameter was selected on)."""
    if "C_sweep" in rec:
        return max(s["macro_f1"] for s in rec["C_sweep"])
    if "alpha_sweep" in rec:
        sweep = rec["alpha_sweep"]
        # Legacy Ridge runs predate ``select_by`` and were tuned on validation
        # mean cosine, the only score their sweep records.
        cosine = rec.get("select_by") == "cosine" or "r2" not in sweep[0]
        return max(s["mean_cosine" if cosine else "r2"] for s in sweep)
    raise KeyError(f"run {rec.get('run_id', '?')} has no validation sweep")


def select_by_val(runs: Iterable[dict]) -> dict:
    """Run with the highest validation score; ties go to the first run given."""
    runs = list(runs)
    if not runs:
        raise ValueError("no runs to select from")
    return max(runs, key=val_score)
