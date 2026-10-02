"""The selection-sensitive cell list (scripts/build_selection_sensitive.py): the
union of the two determinism diffs, each cell with what moved under which
perturbation."""
from __future__ import annotations

import build_selection_sensitive as bs


def _diff(cells, picks=()):
    return {"stamps": {"a": {"git_sha": "a"}, "b": {"git_sha": "b"}}, "cells": cells, "picks": list(picks)}


def _cell(key, what, f1=None, hp=(1.0, 1.0)):
    c = {"key": key, "what": what, "hp_a": hp[0], "hp_b": hp[1], "d": 10, "n_train": 2271}
    if f1 is not None:
        c["metric_deltas"] = {"test_macro_f1": f1}
    return c


def test_the_union_keeps_each_perturbation_apart():
    threads = _diff([_cell("s/cds/family5/a", ["pick", "predictions", "metrics"], 0.03, (1.0, 100.0)),
                     _cell("s/cds/family5/b", ["edge"])])
    kernel = _diff([_cell("s/cds/family5/a", ["predictions", "metrics"], -0.01),
                    _cell("s/cds/genept/gc", ["predictions"])],
                   picks=[{"pick": "s/cds/family5 best encoder", "a": "x", "b": "y"}])
    out = bs.build(threads, kernel)
    by = {c["key"]: c for c in out["cells"]}
    assert set(by) == {"s/cds/family5/a", "s/cds/family5/b", "s/cds/genept/gc"}
    a = by["s/cds/family5/a"]
    assert a["threads"]["what"] == ["pick", "predictions", "metrics"] and a["kernel"]["what"] == ["predictions", "metrics"]
    assert a["pick_changed"] and a["max_abs_d_test_f1"] == 0.03
    assert by["s/cds/family5/b"]["kernel"] is None and not by["s/cds/family5/b"]["pick_changed"]
    assert by["s/cds/genept/gc"]["threads"] is None and by["s/cds/genept/gc"]["max_abs_d_test_f1"] is None
    assert out["builder_picks"] == {"threads": [], "kernel": kernel["picks"]}
    assert out["summary"] == {"cells": 3, "pick_changed": 1, "both_perturbations": 1, "builder_picks": 1}
    assert out["sources"]["kernel"] == {"a": "a", "b": "b"}


def test_a_cell_that_differs_nowhere_is_not_listed():
    out = bs.build(_diff([]), _diff([]))
    assert out["cells"] == [] and out["summary"]["cells"] == 0
