"""The key-aware diff of two recompute runs (scripts/diff_records.py): Phase 4's
1-vs-6-thread determinism check and Phase 5's clean-room comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import diff_records as dr
from data_loader.model_registry import encoder_pools
from linear_trainer.cell import arrays_sha256

POOLS = [f"nt_v2_{p}" for p in encoder_pools("nt_v2", "CDS")]


def _clf(source, C=1.0, pred=("tf", "gpcr"), val=0.5, **over):
    arrays = {"ids": np.array(["g1", "g2"]), "pred": np.array(pred), "y_true": np.array(["tf", "tf"])}
    return {"key": f"splits.json/cds/family5/{source}", "split": "splits.json", "arm": "cds",
            "task": "family5", "feature_source": source, "shuffled_labels": False, "label_seed": None,
            "C": C, "C_sweep": [{"C": C, "macro_f1": val, "converged": True, "n_iter": 10}],
            "edge": False, "feature_dim": 768, "test_macro_f1": 0.5, "converged": True, "n_iter": 10,
            "stamp": {"git_sha": "a" * 40, "threads": 1}, "protocol_hash": "p",
            "pred_sha256": arrays_sha256(arrays), **over}


def _reg(root: Path, source, pred, r2=0.1, alpha=10.0, **over):
    arrays = {"ids": np.array(["g1", "g2"]), "pred": np.asarray(pred, dtype=np.float32)}
    rel = f"pred/{source}__{arrays_sha256(arrays)[:12]}.npz"
    (root / "pred").mkdir(parents=True, exist_ok=True)
    np.savez(root / rel, **arrays)
    return {"key": f"splits.json/cds/genept/{source}", "split": "splits.json", "arm": "cds",
            "task": "genept", "feature_source": source, "shuffled_labels": False, "label_seed": None,
            "alpha": alpha, "alpha_sweep": [{"alpha": alpha, "r2": 0.1, "converged": True}],
            "select_by": "r2", "edge": False, "feature_dim": 768, "test_r2_macro": r2,
            "unpurged": {"test_r2_macro": r2}, "converged": True, "n_iter": 5,
            "stamp": {"git_sha": "a" * 40, "threads": 1}, "protocol_hash": "p",
            "pred_file": rel, "pred_sha256": arrays_sha256(arrays), **over}


def _write(d: Path, recs: list[dict], name="metrics_splits.json") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(recs))
    return d


def _diff(tmp_path, a, b, **kw):
    return dr.diff(_write(tmp_path / "a", a), _write(tmp_path / "b", b),
                   a_root=tmp_path / "a", b_root=tmp_path / "b", **kw)


def test_identical_runs_have_no_differences(tmp_path):
    recs = [_clf("aa2"), _clf("kmer", C=10.0)]
    out = _diff(tmp_path, recs, recs)
    assert out["n_common"] == 2 and not dr.differs(out)


def test_provenance_is_not_compared(tmp_path):
    b = [_clf("aa2", stamp={"git_sha": "b" * 40, "threads": 6}, protocol_hash="q", n_iter=99)]
    out = _diff(tmp_path, [_clf("aa2")], b)
    assert not dr.differs(out)
    assert out["n_iter_changed"] == 1                                      # reported, not a difference
    assert out["stamps"]["b"] == {"git_sha": "b" * 40, "protocol_hash": "q", "threads": [6]}


def test_a_pick_flip_is_flagged_with_the_fragility_tags(tmp_path):
    out = _diff(tmp_path, [_clf("aa2", C=1.0)], [_clf("aa2", C=1000.0, pred=("tf", "tf"))])
    (cell,) = out["cells"]
    assert cell["key"] == "splits.json/cds/family5/aa2" and "pick" in cell["what"]
    assert (cell["hp_a"], cell["hp_b"], cell["d"], cell["n_train"]) == (1.0, 1000.0, 768, 2271)
    assert dr.differs(out)


def test_classification_predictions_must_match_bit_for_bit(tmp_path):
    out = _diff(tmp_path, [_clf("aa2")], [_clf("aa2", pred=("tf", "tf"))])
    (cell,) = out["cells"]
    assert cell["what"] == ["predictions"]


def test_a_classification_metric_must_match_exactly(tmp_path):
    out = _diff(tmp_path, [_clf("aa2")], [_clf("aa2", test_macro_f1=0.5 + 1e-15)])
    assert out["cells"][0]["what"] == ["metrics"]


def test_regression_noise_below_the_tolerance_matches(tmp_path):
    a = [_reg(tmp_path / "a", "aa2", [0.1, 0.2], r2=0.1)]
    b = [_reg(tmp_path / "b", "aa2", [0.1, 0.2], r2=0.1 + 1e-12)]
    assert not dr.differs(_diff(tmp_path, a, b))


def test_regression_drift_above_the_tolerance_is_reported(tmp_path):
    a = [_reg(tmp_path / "a", "aa2", [0.1, 0.2], r2=0.1)]
    b = [_reg(tmp_path / "b", "aa2", [0.1, 0.2001], r2=0.1 + 1e-6)]
    (cell,) = _diff(tmp_path, a, b)["cells"]
    assert cell["what"] == ["predictions", "metrics"]
    assert cell["max_abs_pred"] == pytest.approx(1e-4, rel=1e-2)
    assert cell["metric_deltas"]["test_r2_macro"] == pytest.approx(1e-6)
    assert cell["metric_deltas"]["unpurged.test_r2_macro"] == pytest.approx(1e-6)


def test_a_convergence_flip_is_a_difference(tmp_path):
    out = _diff(tmp_path, [_clf("aa2")], [_clf("aa2", converged=False)])
    assert out["cells"][0]["what"] == ["converged"]


def test_a_missing_key_is_an_error(tmp_path):
    with pytest.raises(dr.BadInput, match="only in a"):
        _diff(tmp_path, [_clf("aa2"), _clf("kmer")], [_clf("aa2")])


def test_a_subset_run_may_omit_keys_but_never_add_them(tmp_path):
    out = _diff(tmp_path, [_clf("aa2"), _clf("kmer")], [_clf("aa2")], subset=True)
    assert out["n_common"] == 1 and out["n_only_a"] == 1 and not dr.differs(out)
    with pytest.raises(dr.BadInput, match="only in b"):
        _diff(tmp_path, [_clf("aa2")], [_clf("aa2"), _clf("kmer")], subset=True)


def test_a_records_file_mixing_two_runs_is_refused(tmp_path):
    a = [_clf("aa2"), _clf("kmer", stamp={"git_sha": "c" * 40, "threads": 1})]
    with pytest.raises(dr.BadInput, match="stamps"):
        _diff(tmp_path, a, a)


def test_a_pool_flip_is_caught_even_when_every_cell_matches_on_its_own(tmp_path):
    """Each cell keeps its C and predictions, but validation scores move enough to
    swap the encoder's pool pick: the cell-level diff alone would miss it."""
    a = [_clf(s, val=0.6 if i == 0 else 0.5) for i, s in enumerate(POOLS)]
    b = [_clf(s, val=0.5 if i == 0 else (0.61 if i == 1 else 0.5)) for i, s in enumerate(POOLS)]
    out = _diff(tmp_path, a, b)
    assert not out["cells"]
    assert {"pick": "splits.json/cds/family5 nt_v2 pool", "a": POOLS[0], "b": POOLS[1]} in out["picks"]
    assert dr.differs(out)
    assert out["max_val_delta"] == pytest.approx(0.11)


def test_identical_pool_picks_report_nothing(tmp_path):
    recs = [_clf(s, val=0.5 + 0.01 * i) for i, s in enumerate(POOLS)]
    out = _diff(tmp_path, recs, recs)
    assert not out["picks"] and not dr.differs(out)


def test_the_cli_exit_codes(tmp_path):
    a = _write(tmp_path / "a", [_clf("aa2")])
    same = _write(tmp_path / "same", [_clf("aa2")])
    other = _write(tmp_path / "other", [_clf("aa2", C=5.0)])
    short = _write(tmp_path / "short", [])
    args = ["--a-root", str(tmp_path), "--b-root", str(tmp_path)]
    assert dr.main([str(a), str(same), *args]) == 0
    assert dr.main([str(a), str(other), *args, "--json", str(tmp_path / "d.json")]) == 1
    assert json.loads((tmp_path / "d.json").read_text())["cells"][0]["hp_b"] == 5.0
    assert dr.main([str(a), str(short), *args]) == 2


def test_an_empty_or_missing_run_is_unusable(tmp_path):
    empty = _write(tmp_path / "e1", [])
    with pytest.raises(dr.BadInput, match="no records"):
        dr.diff(empty, _write(tmp_path / "e2", []))
    with pytest.raises(dr.BadInput, match="no records"):
        dr.diff(_write(tmp_path / "a", [_clf("aa2")]), tmp_path / "typo", subset=True)


def test_a_run_is_never_compared_with_itself(tmp_path):
    a = _write(tmp_path / "a", [_clf("aa2")])
    with pytest.raises(dr.BadInput, match="same directory"):
        dr.diff(a, tmp_path / "a" / ".." / "a")


@pytest.mark.parametrize("field,value", [("edge", "nonconverged"), ("degenerate", True),
                                         ("n_test_scored", 458),
                                         ("n_test_scored_by_class", {"tf": 1}),
                                         ("purge", {"rules": []}), ("targets_sha256", "x")])
def test_a_field_the_statistics_gate_on_is_compared(tmp_path, field, value):
    """build_statistics refuses a confirmatory cell on its edge, degeneracy and
    per-family counts: a change there is a difference even with identical predictions."""
    base = {"edge": False, "degenerate": False, "n_test_scored": 459,
            "n_test_scored_by_class": {"tf": 2}, "purge": {"rules": ["protein@0.40"]},
            "targets_sha256": "t"}
    out = _diff(tmp_path, [_clf("aa2", **base)], [_clf("aa2", **{**base, field: value})])
    assert out["cells"][0]["what"] == [field]


def test_a_metric_on_one_side_only_is_a_difference(tmp_path):
    (cell,) = _diff(tmp_path, [_clf("aa2")], [_clf("aa2", test_kappa=0.3)])["cells"]
    assert cell["what"] == ["metrics"] and cell["metric_deltas"] == {"test_kappa": None}


def test_a_nan_metric_is_a_difference(tmp_path):
    a = [_reg(tmp_path / "a", "aa2", [0.1, 0.2])]
    b = [_reg(tmp_path / "b", "aa2", [0.1, 0.2], r2=float("nan"))]
    assert _diff(tmp_path, a, b)["cells"][0]["what"] == ["metrics"]
    both = [_reg(tmp_path / "b", "aa2", [0.1, 0.2], r2=float("nan"))]
    assert not dr.differs(_diff(tmp_path, both, both))


def test_a_missing_prediction_file_is_unusable(tmp_path):
    a = [_reg(tmp_path / "a", "aa2", [0.1, 0.2])]
    b = [_reg(tmp_path / "b", "aa2", [0.1, 0.3])]
    (tmp_path / "b" / b[0]["pred_file"]).unlink()
    with pytest.raises(dr.BadInput, match="prediction file"):
        _diff(tmp_path, a, b)


def test_the_e5_composition_pick_is_compared(tmp_path):
    def tss(source, val):
        return {**_clf(source, val=val), "key": f"splits_tss_disjoint.json/tss/family5/{source}",
                "split": "splits_tss_disjoint.json", "arm": "tss"}
    names = ["tss_nt_v2_chunk4mergc", "tss_nt_v2_chunk6mer"]
    out = _diff(tmp_path, [tss(names[0], 0.6), tss(names[1], 0.5)],
                [tss(names[0], 0.5), tss(names[1], 0.6)])
    assert {"pick": "splits_tss_disjoint.json/tss/family5 nt_v2 anchored-chunk composition",
            "a": names[0], "b": names[1]} in out["picks"]
