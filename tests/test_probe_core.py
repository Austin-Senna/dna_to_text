"""G6 (grids and edges) and G7 (stamps, threads) for the single fit path."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import train_logistic_probe as tlp
import train_probe as tp
from synth import call_main, read_records, redirect_splits, sha256_file, write_dataset

REPO = Path(__file__).resolve().parents[1]
RECORD_KEYS = {"stamp", "protocol", "protocol_hash", "splits_file", "splits_sha256", "grid",
               "edge", "n_iter", "converged", "pred_file", "pred_sha256", "features", "purge"}
STAMP_KEYS = {"git_sha", "git_dirty", "threads", "machine", "versions"}


# --- G6: the pick never silently sits on a grid edge -------------------------

def test_ridge_pick_is_interior_when_the_optimum_lies_past_the_old_grid(monkeypatch, tmp_path):
    parquet, splits = write_dataset(tmp_path, sizes=(600, 300, 300), d=200, k=16,
                                    ridge_beta2=3e-4)
    redirect_splits(monkeypatch, splits)
    monkeypatch.setattr(tp, "PRED_ROOT", tmp_path / "preds")
    out = tmp_path / "m.json"
    call_main(tp, ["--dataset", str(parquet), "--probe-out", str(tmp_path / "p.npz"),
                   "--metrics-out", str(out)])
    rec = read_records(out)[-1]
    alphas = [s["alpha"] for s in rec["alpha_sweep"]]
    assert min(alphas) < rec["alpha"] < max(alphas), (rec["alpha"], alphas)
    assert rec["alpha"] == 1e4 and rec["edge"] is False


def _sweep(scores, **kw):
    from linear_trainer.fit import extend_sweep
    from linear_trainer.protocol import V2
    p = replace(V2, **kw)
    return extend_sweep(lambda hp: (scores(hp), True), p.base_grid, p.limits["C"],
                        p.plateau_eps)


def test_extension_walks_past_the_base_grid_to_an_interior_peak():
    points, pick, edge = _sweep(lambda hp: -(math.log10(hp) - 5.4) ** 2)
    assert pick == pytest.approx(1e5) and edge is False
    assert max(p["hp"] for p in points) == pytest.approx(1e6)


def test_a_flat_curve_stops_on_a_plateau_flag():
    points, pick, edge = _sweep(lambda hp: -1e-6 / hp)
    assert edge == "plateau" and pick == pytest.approx(1e5)


def test_a_rising_curve_stops_at_the_limit_flag():
    points, pick, edge = _sweep(lambda hp: math.log10(hp))
    assert edge == "limit" and pick == pytest.approx(1e6)


def test_a_tie_at_the_edge_keeps_the_earlier_pick_and_is_flagged():
    # the constant HyenaDNA clsmean case: every C scores 0.140107
    points, pick, edge = _sweep(lambda hp: 0.140107)
    assert pick == pytest.approx(1e-4) and edge == "plateau"
    assert min(p["hp"] for p in points) == pytest.approx(1e-5)


def test_a_curve_that_saturates_at_the_edge_is_flagged():
    points, pick, edge = _sweep(lambda hp: min(math.log10(hp), 4.0))
    assert pick == pytest.approx(1e4) and edge == "plateau"


def test_non_converged_points_are_ineligible_and_stop_extension():
    from linear_trainer.fit import extend_sweep
    from linear_trainer.protocol import V2
    points, pick, edge = extend_sweep(lambda hp: (math.log10(hp), hp < 1e3),
                                      V2.base_grid, V2.limits["C"], V2.plateau_eps)
    assert pick == pytest.approx(100.0) and edge == "nonconverged"
    assert max(p["hp"] for p in points) == pytest.approx(1e4)


def test_a_non_converged_neighbour_flags_an_interior_pick():
    from linear_trainer.fit import extend_sweep
    from linear_trainer.protocol import V2
    scores = {1e3: 9.0, 1e4: 0.0}                    # 1e3 fails; 1e4 converges but is worse
    points, pick, edge = extend_sweep(
        lambda hp: (scores.get(hp, math.log10(hp)), hp != 1e3),
        V2.base_grid, V2.limits["C"], V2.plateau_eps)
    assert pick == pytest.approx(100.0) and edge == "nonconverged"


# --- G7: every record is stamped, and the thread count is asserted -----------

def _cls_record(monkeypatch, tmp_path, dataset="synthetic_cell", task="family5"):
    parquet, splits = write_dataset(tmp_path)
    monkeypatch.setitem(tlp.DATASET_PATHS, "synthetic_cell", parquet)
    monkeypatch.setattr(tlp, "PRED_ROOT", tmp_path / "preds")
    out = tmp_path / "m.json"
    call_main(tlp, ["--dataset", dataset, "--task", task, "--splits", str(splits),
                    "--metrics-out", str(out)])
    return read_records(out)[-1], splits


def _check_stamp(rec, split_file):
    assert RECORD_KEYS <= rec.keys(), sorted(RECORD_KEYS - rec.keys())
    assert STAMP_KEYS <= rec["stamp"].keys()
    assert rec["stamp"]["threads"] == 1
    assert Path(REPO / rec["splits_file"]).resolve() == Path(split_file).resolve()
    assert rec["splits_sha256"] == sha256_file(split_file)
    from linear_trainer.cell import load_predictions
    load_predictions(rec)            # raises if the stored arrays don't match pred_sha256


def test_classification_records_carry_the_full_stamp(monkeypatch, tmp_path):
    _check_stamp(*_cls_record(monkeypatch, tmp_path))


def test_binary_records_stamp_their_own_subset_file(monkeypatch, tmp_path):
    rec, _ = _cls_record(monkeypatch, tmp_path, dataset="nt_v2_specialmean", task="tf_vs_gpcr")
    _check_stamp(rec, REPO / "data" / "binary_tf_vs_gpcr.json")


def test_regression_records_carry_the_full_stamp(monkeypatch, tmp_path):
    parquet, splits = write_dataset(tmp_path)
    monkeypatch.setattr(tp, "PRED_ROOT", tmp_path / "preds")
    out = tmp_path / "m.json"
    call_main(tp, ["--dataset", str(parquet), "--probe-out", str(tmp_path / "p.npz"),
                   "--splits", str(splits), "--metrics-out", str(out)])
    rec = read_records(out)[-1]
    _check_stamp(rec, splits)
    assert "feature_source" not in rec and "task" not in rec   # builders filter on these


def test_an_unpinned_thread_pool_is_refused(monkeypatch, tmp_path):
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [
        {"user_api": "blas", "internal_api": "openblas", "num_threads": 12}])
    with pytest.raises(RuntimeError, match="thread"):
        _cls_record(monkeypatch, tmp_path)


# --- the fit itself ------------------------------------------------------------

def test_scaled_predict_matches_the_sklearn_pipeline_on_a_badly_scaled_real_cell():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from linear_trainer.fit import fit
    from linear_trainer.protocol import V2
    from splits import load_split

    X, _, meta = load_split("train", dataset_path=REPO / "data/dataset_hyena_dna_clsmean.parquet",
                            splits_path=REPO / "data/splits.json")
    X, y = X.astype(np.float64), meta["family"].to_numpy()
    probe = fit("logistic", X, y, 1.0, V2)
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, tol=V2.tol,
                                                             max_iter=V2.max_iter)).fit(X, y)
    np.testing.assert_allclose(probe.decision_function(X), ref.decision_function(X),
                               rtol=1e-8, atol=1e-8)
    assert (probe.predict(X) == ref.predict(X)).all()


def test_a_refit_that_does_not_converge_raises():
    from linear_trainer.fit import ConvergenceFailure, fit
    from linear_trainer.protocol import V2
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    y = np.array(["a", "b"] * 100)
    with pytest.raises(ConvergenceFailure):
        fit("logistic", X, y, 1e4, replace(V2, max_iter=2))
