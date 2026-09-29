"""Validation-selected headline cells, derived from the primary homology-split metrics.

Single source of truth for which cells the robustness scripts (bootstrap CIs,
split-seed sensitivity, 70%-identity split) re-probe: each encoder's pooling is
the one with the best VALIDATION score (``linear_trainer.selection``, the same
rule ``build_paper_tables.py`` uses), never the best test score, at its recorded
validation-selected C / alpha.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_loader.pooling_aggregator import POOLING_VARIANTS
from linear_trainer.selection import select_pool

DATA = Path(__file__).resolve().parents[1] / "data"
ENCODERS = ("dnabert2", "nt_v2", "gena_lm", "hyena_dna")
HOMOLOGY = json.loads((DATA / "metrics_homology.json").read_text())


def reg_source(rec: dict) -> str | None:
    """Source id for a regression record (mirrors build_paper_tables.reg_raw)."""
    if rec.get("model") == "linear_probe" and rec.get("dataset"):
        return rec["dataset"].replace("dataset_", "").replace(".parquet", "")
    return rec.get("feature_source")


# source id -> record; later records win, as in build_paper_tables.
CLS_RECS = {r["feature_source"]: r for r in HOMOLOGY
            if r.get("task") == "family5" and not r.get("shuffled_labels")}
REG_RECS = {reg_source(r): r for r in HOMOLOGY
            if r.get("task") is None and "test_r2_macro" in r and reg_source(r)}


def best(recs: dict[str, dict], prefix: str, pools=POOLING_VARIANTS) -> str:
    """Validation-selected source among ``prefix + pool`` for the named pools.

    The candidates are named explicitly, not matched by prefix, so E5 cells
    (``tssanchored``, ``centermean``) never join the pick (ledger G14).
    """
    return select_pool(recs, [prefix + p for p in pools])


CLS_BEST = {e: best(CLS_RECS, e + "_") for e in ENCODERS}
REG_BEST = {e: best(REG_RECS, e + "_") for e in ENCODERS}
CLS_BEST_TSS = {e: best(CLS_RECS, f"tss_{e}_") for e in ENCODERS}
REG_BEST_TSS = {e: best(REG_RECS, f"tss_{e}_") for e in ENCODERS}
# Best CDS DNA-encoder cell across encoders, again chosen on validation.
BEST_DNA_CLS = select_pool(CLS_RECS, [CLS_BEST[e] for e in ENCODERS])
BEST_DNA_REG = select_pool(REG_RECS, [REG_BEST[e] for e in ENCODERS])
