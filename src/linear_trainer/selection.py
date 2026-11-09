"""Validation-based model selection over recorded probe runs.

Every probe run records its hyperparameter sweep on the validation split
(``C_sweep`` for logistic probes, ``alpha_sweep`` for Ridge). Choosing a pooling
rule (or any other configuration) for an encoder must use those validation
scores, never the test metrics, so the held-out test split is scored exactly
once per reported cell.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class MissingRecord(RuntimeError, LookupError):
    """A cell, pool or hyperparameter the caller needs has no record."""


def _converged(sweep: list[dict]) -> list[dict]:
    # Rows without the flag predate convergence tracking and count as converged.
    return [s for s in sweep if s.get("converged", True)]


def val_score(rec: dict) -> float:
    """Best validation score in a run's sweep (the value its hyperparameter was selected on).

    Points that did not converge were never eligible for the pick, so they don't count.
    """
    if "C_sweep" in rec:
        return max(s["macro_f1"] for s in _converged(rec["C_sweep"]))
    if "alpha_sweep" in rec:
        sweep = _converged(rec["alpha_sweep"])
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


def select_pool(records: Mapping[str, dict], candidates: Sequence[str]) -> str:
    """The validation-selected source among an explicit candidate list.

    Candidates are named, never matched by prefix, so a record that merely
    shares a name stem (an E5 ``tss_*_tssanchored`` or ``centermean`` cell)
    can't join the pick (ledger G14). Ties go to the earlier candidate, so the
    pick doesn't depend on record order.
    """
    present = [c for c in candidates if c in records]
    if not present:
        raise MissingRecord(f"none of the candidates has a record: {list(candidates)}")
    return max(present, key=lambda c: val_score(records[c]))


def encoder_cells(encoder: str, pools: Sequence[str] | None = None) -> frozenset[str]:
    """The sources that count as one encoder's cells: its base parquet and its named pools.

    Builders filter records by membership in this set instead of a name prefix
    (ledger G14), so ``<encoder>_tssanchored``-style cells never join a pick.
    The default pools are the encoder's own (``model_registry.encoder_pools``),
    so HyenaDNA's dropped clsmean/specialmean records can never be candidates.
    """
    if pools is None:
        from data_loader.model_registry import ENCODER_SPECS, encoder_pools
        from data_loader.pooling_aggregator import POOLING_VARIANTS
        pools = encoder_pools(encoder) if encoder in ENCODER_SPECS else POOLING_VARIANTS
    return frozenset([encoder, *(f"{encoder}_{p}" for p in pools)])


def legacy_alpha(rec: dict) -> float:
    """The alpha of a record fitted by the May protocol (unscaled Ridge).

    Scripts that still refit a raw Ridge at a recorded alpha can reproduce only
    those records. A record fitted under a newer protocol carries
    ``protocol_hash``: its alpha was chosen on standardised features, so a raw
    refit would silently give different numbers. Those records must be rescored
    from their stored predictions instead (Phase 1D).
    """
    if "protocol_hash" in rec:
        raise RuntimeError(f"{rec.get('run_id', '?')} was fitted under protocol {rec.get('protocol')}; "
                           "rescore its stored predictions instead of refitting a raw Ridge")
    return float(rec["alpha"])
