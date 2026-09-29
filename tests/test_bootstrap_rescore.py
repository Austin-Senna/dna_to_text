"""G8: bootstraps rescore the stored test predictions of the recorded fit.

A refit can land on a different model (another machine, thread count or
max_iter), so the bootstrap point then no longer equals the table value. The
bootstrap must not fit at all, and its point must equal the record.
"""
from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression, Ridge

import bootstrap_test_uncertainty as bt
import train_logistic_probe as tlp
import train_probe as tp
from synth import call_main, read_records, write_dataset


def _no_fits(monkeypatch):
    def refuse(self, *a, **k):
        raise AssertionError("bootstrap refitted a probe")
    monkeypatch.setattr(LogisticRegression, "fit", refuse)
    monkeypatch.setattr(Ridge, "fit", refuse)


def _other_features(parquet, out):
    """Same genes, split and GenePT targets (as in the real parquets), different X."""
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(parquet)
    rng = np.random.default_rng(1)
    df["x"] = [np.asarray(x, dtype=np.float32) + rng.standard_normal(len(x)).astype(np.float32)
               for x in df["x"]]
    df.to_parquet(out)
    return out


def _records(monkeypatch, tmp_path):
    parquet, splits = write_dataset(tmp_path)
    other = _other_features(parquet, tmp_path / "other.parquet")
    monkeypatch.setitem(tlp.DATASET_PATHS, "synthetic_cell", parquet)
    monkeypatch.setitem(tlp.DATASET_PATHS, "other_cell", other)
    for mod in (tlp, tp):
        monkeypatch.setattr(mod, "PRED_ROOT", tmp_path / "preds")
    out = tmp_path / "m.json"
    for ds in ("synthetic_cell", "other_cell"):
        call_main(tlp, ["--dataset", ds, "--task", "family5", "--splits", str(splits),
                        "--metrics-out", str(out)])
    for pq in (parquet, other):
        call_main(tp, ["--dataset", str(pq), "--probe-out", str(tmp_path / "p.npz"),
                       "--splits", str(splits), "--metrics-out", str(out)])
    return read_records(out)


def test_classification_bootstrap_rescores_without_refitting(monkeypatch, tmp_path):
    cls, _, _, _ = _records(monkeypatch, tmp_path)
    _no_fits(monkeypatch)
    res = bt.bootstrap_classification(cls, n_iters=50)
    lo, hi = res["macro_f1_ci95"]
    assert res["macro_f1_point"] == cls["test_macro_f1"] and lo <= hi


def test_regression_bootstrap_rescores_without_refitting(monkeypatch, tmp_path):
    _, _, reg, _ = _records(monkeypatch, tmp_path)
    _no_fits(monkeypatch)
    res = bt.bootstrap_regression(reg, n_iters=50)
    assert res["r2_macro_point"] == reg["test_r2_macro"]


@pytest.mark.parametrize("which,key", [(0, "test_macro_f1"), (2, "test_r2_macro")])
def test_a_record_whose_value_the_predictions_do_not_reproduce_is_refused(
        monkeypatch, tmp_path, which, key):
    rec = _records(monkeypatch, tmp_path)[which]
    boot = bt.bootstrap_classification if which == 0 else bt.bootstrap_regression
    with pytest.raises(RuntimeError, match=key):
        boot({**rec, key: rec[key] + 1e-9}, n_iters=5)


def test_paired_point_equals_the_difference_of_two_cells(monkeypatch, tmp_path):
    a, b, ra, rb = _records(monkeypatch, tmp_path)
    _no_fits(monkeypatch)
    res = bt.paired_bootstrap_classification(a, b, n_iters=20)
    assert res["n_common"] == 60
    assert res["delta_macro_f1_point"] == pytest.approx(a["test_macro_f1"] - b["test_macro_f1"], abs=1e-12)
    assert res["delta_macro_f1_point"] != 0.0
    res = bt.paired_bootstrap_regression(ra, rb, n_iters=20)
    assert res["delta_r2_macro_point"] == pytest.approx(ra["test_r2_macro"] - rb["test_r2_macro"], abs=1e-9)


def test_a_tampered_prediction_file_is_refused(monkeypatch, tmp_path):
    import numpy as np
    from linear_trainer.cell import load_predictions
    cls = _records(monkeypatch, tmp_path)[0]
    arrays = dict(np.load(cls["pred_file"], allow_pickle=False))
    arrays["pred"] = arrays["pred"][::-1].copy()
    np.savez(cls["pred_file"], **arrays)
    with pytest.raises(RuntimeError, match="sha256"):
        load_predictions(cls)
