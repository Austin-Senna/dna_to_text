"""The triplicated-Mean control (D5, 3x C): what it loads, that it is Mean at 3x C,
and that it never enters selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loader.model_registry import encoder_pools
from linear_trainer import records as R
from linear_trainer import sources
from linear_trainer.cell import _features_stamp
from linear_trainer.fit import fit
from linear_trainer.protocol import V2

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits.json"


def test_each_encoder_has_one_control_built_on_its_mean():
    assert sources.DERIVED == {f"{e}_meanmean3": (f"{e}_meanmean", 3) for e in sources.ENCODERS}
    for e in sources.ENCODERS:
        assert "meanmean" in encoder_pools(e, "CDS")


@pytest.mark.parametrize("task", ["family5", "genept"])
def test_the_control_is_mean_copied_three_times(task):
    X, y, ids = sources.load("nt_v2_meanmean", task, "val", SPLITS)
    X3, y3, ids3 = sources.load("nt_v2_meanmean3", task, "val", SPLITS)
    assert np.array_equal(ids3, ids) and np.array_equal(y3, y)
    assert np.array_equal(X3, np.hstack([X, X, X]))


def test_the_control_stamp_names_its_base_and_transform():
    stamp = _features_stamp("nt_v2_meanmean3")
    assert stamp["path"] == _features_stamp("nt_v2_meanmean")["path"]
    assert (stamp["derived_from"], stamp["transform"]) == ("nt_v2_meanmean", "hstack x3")


def test_the_control_is_never_a_selection_candidate():
    for e in sources.ENCODERS:
        assert "meanmean3" not in encoder_pools(e, "CDS")
    # A control with the best validation score of all still cannot be picked.
    def rec(f1):
        return {"C": 1.0, "C_sweep": [{"C": 1.0, "macro_f1": f1, "converged": True}]}
    recs = {f"dnabert2_{p}": rec(0.5) for p in encoder_pools("dnabert2", "CDS")}
    recs["dnabert2_meanmean3"] = rec(0.99)
    assert R.best_pool(recs, "dnabert2", "cds") != "dnabert2_meanmean3"


def _data(seed=0, n=240, d=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 5, size=d) + rng.normal(size=d)
    w = rng.normal(size=(d, 3))
    y = np.argmax(X @ w + rng.normal(scale=2.0, size=(n, 3)), axis=1)
    Y = X @ rng.normal(size=(d, 5)) + rng.normal(size=(n, 5))
    return X, y, Y


@pytest.mark.parametrize("factor,equal", [(3, True), (2, False)])
def test_logistic_on_three_copies_is_mean_at_three_times_c(factor, equal):
    X, y, _ = _data()
    X3 = np.hstack([X, X, X])
    for C in (0.01, 0.1, 1.0):
        s3 = fit("logistic", X3, y, C, V2).decision_function(X3)
        s1 = fit("logistic", X, y, factor * C, V2).decision_function(X)
        assert np.allclose(s3, s1, atol=1e-3) is equal, C


@pytest.mark.parametrize("factor,equal", [(3, True), (2, False)])
def test_ridge_on_three_copies_is_mean_at_a_third_of_alpha(factor, equal):
    _, _, Y = _data()
    X, _, _ = _data(seed=1)
    X3 = np.hstack([X, X, X])
    for alpha in (1.0, 100.0, 1e4):
        p3 = fit("ridge", X3, Y, alpha, V2).predict(X3)
        p1 = fit("ridge", X, Y, alpha / factor, V2).predict(X)
        assert np.allclose(p3, p1, atol=1e-8) is equal, alpha
