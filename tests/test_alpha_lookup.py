"""G16: a downstream refit never falls back to a made-up alpha."""
from __future__ import annotations

import pytest


def test_per_dim_r2_refuses_a_missing_record():
    import per_dim_r2
    with pytest.raises(LookupError):
        per_dim_r2._alpha_for("dataset_not_a_cell.parquet")


def test_ridge_robust_refuses_a_missing_record():
    import ridge_robust_metrics
    with pytest.raises(LookupError):
        ridge_robust_metrics.alpha_for("dataset_not_a_cell.parquet", [])


@pytest.mark.parametrize("stamped", [False, True])
def test_a_raw_ridge_refit_accepts_only_may_protocol_records(stamped):
    import ridge_robust_metrics
    rec = {"model": "linear_probe", "dataset": "dataset_x.parquet", "alpha": 10.0}
    if stamped:
        rec |= {"protocol": "v2-2026-09", "protocol_hash": "abc"}
        with pytest.raises(RuntimeError, match="rescore"):
            ridge_robust_metrics.alpha_for("dataset_x.parquet", [rec])
    else:
        assert ridge_robust_metrics.alpha_for("dataset_x.parquet", [rec]) == 10.0
