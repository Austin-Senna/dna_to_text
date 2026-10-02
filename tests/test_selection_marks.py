"""Selection-sensitivity marks (Phase 6; Hayden, Oct 2: mark a printed number only
when its digits move under either perturbation of the frozen protocol). The
builder replays each perturbation's metric deltas and builder-pick flips from
``data/v2/selection_sensitive.json``, rebuilds every fragment, and marks the
numbers whose text differs."""
from __future__ import annotations

import json
import re

import pytest

import build_paper_tables as bt
from linear_trainer import records as R

M = bt.MARK


def test_a_number_is_marked_only_when_its_printed_digits_move():
    canon = r"A & 0.660 & +0.012 & 0.300 \\"
    out = bt.mark_unstable(canon, [r"A & 0.661 & +0.012 & 0.300 \\", r"A & 0.660 & -0.001 & 0.300 \\"])
    assert out == rf"A & 0.660{M} & +0.012{M} & 0.300 \\"
    assert bt.mark_unstable(canon, [canon]) == canon


def test_a_change_outside_the_numbers_is_never_marked_silently():
    with pytest.raises(bt.StructureChanged):
        bt.mark_unstable(r"NT-v2 & Mean & 0.660 \\", [r"NT-v2 & Ends & 0.660 \\"])
    with pytest.raises(bt.StructureChanged):     # a replay that adds a row
        bt.mark_unstable("A & 0.660\n", ["A & 0.660\nB & 0.1\n"])


def _sens(cells=(), picks=()):
    return {"sources": {"threads": {"a": "sha"}}, "inputs": {},
            "cells": [{"key": k, "threads": {"hp": [1.0, 1.0], "metric_deltas": {}}} for k in cells],
            "builder_picks": {"threads": [{"pick": n, "a": "x", "b": "y"} for n in picks]}}


@pytest.mark.parametrize("sens,why", [
    (_sens(cells=["splits_nowhere.json/cds/family5/kmer"]), "in no records file"),
    (_sens(picks=["splits_seed1.json/cds/family5 best-encoder"]), "not a pick name"),
    (_sens(picks=["splits.json/cds/family5 best encoder"]), "behind statistics.json"),
    (_sens(picks=["splits_tss_disjoint.json/tss/family5 gena_lm pool"]), "behind statistics.json"),
    (_sens(cells=["splits.json/cds/genept/dnabert2_specialmean"]), "ridge_robust.json"),
])
def test_a_replay_the_tables_cannot_carry_refuses_to_build(sens, why):
    """Review findings 1-2 (Oct 2): an entry that would be dropped, or a flip whose
    statistics pairs cannot be replayed, stops the build instead of passing unmarked."""
    loaded = {"splits.json/cds/family5/kmer", "splits.json/cds/genept/dnabert2_specialmean"}
    with pytest.raises(bt.NotReplayable, match=why):
        bt.check_replayable(sens, "threads", loaded, {"rows": [{"key": "splits.json/cds/genept/dnabert2_specialmean"}]})
    bt.check_replayable(_sens(cells=["splits.json/cds/family5/kmer"],
                              picks=["splits_seed1.json/cds/family5 nt_v2 pool"]), "threads", loaded, {"rows": []})


def test_a_list_built_against_other_records_is_refused(monkeypatch):
    sens = json.loads((R.V2 / "selection_sensitive.json").read_text())
    stale = {**sens, "inputs": {**sens["inputs"], "metrics_splits.json": "0" * 64}}
    monkeypatch.setattr(bt, "_read_sens", lambda: stale)
    with pytest.raises(R.MixedRecords, match="metrics_splits.json"):
        bt.load_records("threads")


def _rec(key, task="family5", **m):
    return {"key": key, "task": task, "C": 1.0, "alpha": 10.0, "unpurged": {"test_macro_f1": 0.5}, **m}


def test_perturbed_records_carry_every_delta_and_leave_the_canonical_alone():
    recs = {"s/cds/family5/a": _rec("s/cds/family5/a", test_macro_f1=0.6, test_kappa=0.5),
            "s/cds/family5/b": _rec("s/cds/family5/b", test_macro_f1=0.7)}
    side = {"hp": [1.0, 100.0], "metric_deltas": {"test_macro_f1": 0.01, "test_kappa": -0.02,
                                                   "unpurged.test_macro_f1": 0.03}}
    out = bt.perturb_records(recs, {"s/cds/family5/a": side})
    a = out["s/cds/family5/a"]
    assert (a["test_macro_f1"], a["test_kappa"], a["unpurged"]["test_macro_f1"], a["C"]) == \
        pytest.approx((0.61, 0.48, 0.53, 100.0))
    assert recs["s/cds/family5/a"]["test_macro_f1"] == 0.6 and recs["s/cds/family5/a"]["unpurged"]["test_macro_f1"] == 0.5
    assert out["s/cds/family5/b"] is recs["s/cds/family5/b"]
    with pytest.raises(R.MixedRecords):          # the list was built against another canonical C
        bt.perturb_records(recs, {"s/cds/family5/a": {**side, "hp": [10.0, 100.0]}})


def test_statistics_points_follow_their_records_and_an_unrebuildable_one_raises():
    before = {"s/cds/family5/a": _rec("s/cds/family5/a", test_macro_f1=0.6, test_kappa=0.4),
              "s/cds/family5/b": _rec("s/cds/family5/b", test_macro_f1=0.5)}
    after = {**before, "s/cds/family5/a": {**before["s/cds/family5/a"], "test_macro_f1": 0.65, "test_kappa": 0.3}}
    stats = {"intervals": {"i": {"key": "s/cds/family5/a", "metric": "test_macro_f1", "point": 0.6,
                                 "kappa_point": 0.4, "ci95": [0.5, 0.7]}},
             "exploratory": {"e": {"a": "s/cds/family5/a", "b": "s/cds/family5/b", "delta_point": 0.1}},
             "null_bands": {}}
    out = bt.perturb_stats(stats, before, after)
    assert out["intervals"]["i"]["point"] == 0.65 and out["intervals"]["i"]["kappa_point"] == 0.3
    assert out["intervals"]["i"]["ci95"] == [0.5, 0.7]               # brackets are not replayed
    assert out["exploratory"]["e"]["delta_point"] == pytest.approx(0.15)
    assert stats["exploratory"]["e"]["delta_point"] == 0.1
    masked = {**stats, "exploratory": {"e": {"a": "s/cds/family5/a", "b": "s/cds/family5/b", "delta_point": 0.09}}}
    with pytest.raises(bt.NotReplayable):        # e.g. a test on masked genes: not a - b of the records
        bt.perturb_stats(masked, before, after)


def test_a_flipped_builder_pick_reaches_the_table(monkeypatch):
    monkeypatch.setattr(bt.R, "best_encoder", lambda by, arm: "gena_lm_meanG")
    split = bt.Split.__new__(bt.Split)
    split.name, split.recs, split.picks = "s.json", {}, {}
    assert split.best_encoder("cds", "family5") == "gena_lm_meanG"
    split.picks = {"s.json/cds/family5 best encoder": ("gena_lm_meanG", "nt_v2_meanD")}
    assert split.best_encoder("cds", "family5") == "nt_v2_meanD"
    split.picks = {"s.json/cds/family5 best encoder": ("dnabert2_meanG", "nt_v2_meanD")}
    with pytest.raises(R.MixedRecords):          # the flip was recorded against another canonical pick
        split.best_encoder("cds", "family5")


def test_every_full_table_cell_that_moves_at_its_printed_precision_is_marked():
    """Real data, checked without the builder's own diff: each family5 cell of the
    full appendix table (4 dp) carries a mark exactly when some perturbation's
    delta changes its printed digits."""
    sens = json.loads((R.V2 / "selection_sensitive.json").read_text())
    side = {p: {c["key"]: c[p]["metric_deltas"] for c in sens["cells"] if c[p]} for p in ("threads", "kernel")}
    marked = bt.build_marked()["pooling_combined"]
    bt.load_records()
    rows = [line for line in marked.splitlines()
            if line.endswith(r"\\") and line.count(" & ") == 7 and not line.startswith("Source & ")]
    items = [it for it in bt._cell_order() if it[0] == "row"]
    assert len(rows) == len(items)
    n_marked = 0
    for (_, _, _, src, arm), row in zip(items, rows):
        cols = row.removesuffix(r" \\").split(" & ")
        for split, vals in ((bt.HOM if arm == "cds" else bt.DIS, cols[2:5]), (bt.RND, cols[5:8])):
            rec = split.cell(arm, "family5", src)
            for metric, txt in zip((bt.F1, bt.K, bt.ACC), vals):
                v = rec[metric]
                moves = any(bt.f(v) != bt.f(v + d[rec["key"]].get(metric, 0.0))
                            for d in side.values() if rec["key"] in d)
                assert (M in txt) is moves, (rec["key"], metric, txt)
                n_marked += moves
    assert n_marked > 0                            # the real list does move printed digits
    assert not re.search(r"\\sens\{\}\\sens\{\}", marked)


def _replayed(key: str, metric: str, value: float, sens: dict) -> list[float]:
    return [value + c[p]["metric_deltas"].get(metric, 0.0)
            for c in sens["cells"] if c["key"] == key for p in ("threads", "kernel") if c[p]]


def test_the_null_median_and_a_flipped_seed_pick_are_marked_on_real_data():
    """Review finding 7: the null-band and builder-pick replays, checked against a
    direct recomputation rather than the builder's own diff."""
    import numpy as np
    sens = json.loads((R.V2 / "selection_sensitive.json").read_text())
    marked = bt.build_marked()
    bt.load_records()
    # TSS null median (3 dp): recomputed from the 200 shuffles with each side's deltas.
    null = [r for r in R.load(bt.TSS, null=True).values()
            if (r["task"], r["feature_source"]) == ("family5", bt.TSS_4MER)]
    canon = bt.f(float(np.median([r[bt.F1] for r in null])), 3)
    moves = False
    for p in ("threads", "kernel"):
        d = {c["key"]: c[p]["metric_deltas"].get(bt.F1, 0.0) for c in sens["cells"] if c[p]}
        moves |= bt.f(float(np.median([r[bt.F1] + d.get(r["key"], 0.0) for r in null])), 3) != canon
    row = next(line for line in marked["s_tss_disjoint"].splitlines() if line.startswith("Shuffled labels"))
    assert (f"{canon}{M}" in row) is moves and moves      # the kernel side moves it (0.1675 -> 0.1678)
    # Seed range of the best DNA encoder: the seed1 pick flips per side.
    def best(split, src, p=None):
        rec = split.cell("cds", "family5", src)
        if p is None:
            return rec[bt.F1]
        c = next((c for c in sens["cells"] if c["key"] == rec["key"] and c[p]), None)
        return rec[bt.F1] + (c[p]["metric_deltas"].get(bt.F1, 0.0) if c else 0.0)
    splits = [bt.HOM, *(c for c, _ in bt.SEED_SPLITS.values())]
    lo = bt.f(min(best(s, s.best_encoder("cds", "family5")) for s in splits), 3)
    lo_moves = False
    for p in ("threads", "kernel"):
        flips = {d["pick"]: d["b"] for d in sens["builder_picks"][p]}
        vals = [best(s, flips.get(f"{s.name}/cds/family5 best encoder", s.best_encoder("cds", "family5")), p)
                for s in splits]
        lo_moves |= bt.f(min(vals), 3) != lo
    row = next(line for line in marked["s_seed_sensitivity"].splitlines() if line.startswith("Best DNA encoder"))
    assert (f"${lo}{M}$" in row) is lo_moves
