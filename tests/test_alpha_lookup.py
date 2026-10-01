"""G16: downstream analyses rescore recorded cells; a missing record raises."""
from __future__ import annotations

import pytest

from linear_trainer import records as R
from linear_trainer.selection import MissingRecord


@pytest.fixture
def no_records(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "V2", tmp_path)


def test_per_dim_r2_refuses_missing_records(no_records):
    import per_dim_r2
    with pytest.raises(MissingRecord):
        per_dim_r2.cells()


def test_ridge_robust_refuses_missing_records(no_records):
    import ridge_robust_metrics
    with pytest.raises(MissingRecord):
        ridge_robust_metrics.cells()


def test_ridge_robust_refuses_predictions_that_do_not_reproduce_the_record(tmp_path):
    import ridge_robust_metrics
    from linear_trainer.cell import run_cell
    from linear_trainer.protocol import V2
    from synth import write_dataset
    parquet, splits = write_dataset(tmp_path)
    res = run_cell(parquet, "genept", splits, V2, pred_dir=tmp_path / "p")
    rec = {**res["provenance"], **res["metrics"], "key": "toy", "alpha": res["hp"]}
    rec["test_r2_macro"] += 1e-9
    with pytest.raises(RuntimeError, match="rescored"):
        ridge_robust_metrics.rescore("toy", rec)


@pytest.mark.parametrize("stamped", [False, True])
def test_a_raw_ridge_refit_accepts_only_may_protocol_records(stamped):
    from linear_trainer.selection import legacy_alpha
    rec = {"model": "linear_probe", "dataset": "dataset_x.parquet", "alpha": 10.0}
    if stamped:
        rec |= {"protocol": "v2-2026-09", "protocol_hash": "abc"}
        with pytest.raises(RuntimeError, match="rescore"):
            legacy_alpha(rec)
    else:
        assert legacy_alpha(rec) == 10.0
