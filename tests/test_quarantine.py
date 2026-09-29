"""G1: selection never sees test labels.

Test labels load only in the final scoring step, after the last fit, and
permuting them leaves the validation sweep and the pick unchanged.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import train_logistic_probe as tlp
from synth import call_main, read_records, redirect_splits, write_dataset


def _prepare(monkeypatch, tmp_path, **kw):
    parquet, splits = write_dataset(tmp_path, **kw)
    monkeypatch.setitem(tlp.DATASET_PATHS, "synthetic_cell", parquet)
    monkeypatch.setattr(tlp, "PRED_ROOT", tmp_path / "preds")
    return splits


def test_test_split_is_read_only_after_the_last_fit(monkeypatch, tmp_path):
    splits = _prepare(monkeypatch, tmp_path)
    log: list[str] = []
    redirect_splits(monkeypatch, splits, log)
    real_fit = LogisticRegression.fit

    def spy(self, *a, **k):
        log.append("fit")
        return real_fit(self, *a, **k)

    monkeypatch.setattr(LogisticRegression, "fit", spy)
    call_main(tlp, ["--dataset", "synthetic_cell", "--task", "family5",
                    "--metrics-out", str(tmp_path / "m.json")])

    last_fit = max(i for i, e in enumerate(log) if e == "fit")
    first_test = log.index("test")
    assert first_test > last_fit, f"test split read before the last fit: {log}"


def test_permuted_test_labels_leave_the_selection_unchanged(monkeypatch, tmp_path):
    runs = []
    for permute in (False, True):
        sub = tmp_path / str(permute)
        splits = _prepare(monkeypatch, sub, permute_test_labels=permute)
        redirect_splits(monkeypatch, splits)
        out = sub / "m.json"
        call_main(tlp, ["--dataset", "synthetic_cell", "--task", "family5",
                        "--metrics-out", str(out)])
        runs.append(read_records(out)[-1])
    real, permuted = runs
    assert real["C_sweep"] == permuted["C_sweep"]
    assert real["C"] == permuted["C"]
    assert real["test_macro_f1"] != permuted["test_macro_f1"]
