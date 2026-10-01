"""G15, G17, G7 and G1 for the recompute runner (scripts/recompute_all.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import recompute_all as ra
from linear_trainer import sources
from splits.leaks import rules_for

DATA = Path(__file__).resolve().parents[1] / "data"

# Pinned so a cell can't silently drop out of the canonical run. Changing the
# manifest means changing these numbers in the same commit, on purpose.
MAIN_CELLS = 754
NULL_CELLS = 1000


def test_the_manifest_size_is_pinned():
    assert len(ra.manifest("main")) == MAIN_CELLS
    assert len(ra.manifest("null")) == NULL_CELLS
    assert len(ra.manifest("all")) == MAIN_CELLS + NULL_CELLS


def test_every_primary_split_has_the_full_grids():
    keys = {c.key for c in ra.manifest("main")}
    for task in ra.TASKS:
        for src in ra.cds_sources():
            assert f"splits.json/cds/{task}/{src}" in keys
        for src in ra.tss_sources() + ra.e5_sources():
            assert f"splits_tss_disjoint.json/tss/{task}/{src}" in keys
    assert "splits_tss_disjoint.json/cds/family5/esm2_650m" in keys      # paired CDS-TSS test
    for src in ra.tss_sources():                                         # the split comparison
        assert f"splits_random.json/tss/family5/{src}" in keys


def test_the_controls_run_on_the_cds_primary_only():
    controls = [c for c in ra.manifest("main") if c.source in sources.DERIVED]
    assert sorted((c.task, c.source) for c in controls) == sorted(
        (t, f"{e}_meanmean3") for t in ra.TASKS for e in ra.ENCODERS)
    assert {(c.split, c.arm) for c in controls} == {(ra.CDS_PRIMARY, "cds")}


def test_every_cell_names_a_split_file_with_a_purge_rule():
    for cell in ra.manifest("all"):
        assert (DATA / cell.split).exists(), cell.split
        rules_for(cell.split, cell.arm)                                      # raises if unnamed


def test_every_source_is_registered():
    for cell in ra.manifest("all"):
        assert (cell.source in sources.SYNTHETIC_FEATURIZERS or cell.source in sources.DATASET_PATHS
                or cell.source in sources.DERIVED), cell.key


@pytest.mark.slow
def test_every_parquet_the_manifest_reads_is_on_disk():
    paths = {sources._parquet_for(c.source) for c in ra.manifest("all")
             if c.source not in sources.SYNTHETIC_FEATURIZERS}
    missing = sorted(str(p) for p in paths if not p.exists())
    assert not missing, missing


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["recompute_all.py", *argv])
    ra.main()


def test_a_failing_cell_fails_the_run(monkeypatch, tmp_path):
    def boom(cell, pred_root):
        raise SystemExit(0)                      # the May runners counted this as success (G17)

    monkeypatch.setattr(ra, "run_one", boom)
    with pytest.raises(RuntimeError, match="cell failed"):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path)])


def test_a_cell_without_its_record_fails_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "run_one", lambda cell, pred_root: {"key": "something else",
                                                               "stamp": {"git_sha": "x"}, "C": 1.0,
                                                               "edge": False, "protocol_hash": "y"})
    with pytest.raises(ra.IncompleteRun):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path)])


def test_records_from_another_commit_are_refused(monkeypatch, tmp_path):
    out = tmp_path / "metrics_splits.json"
    out.write_text(json.dumps([{"key": "k", "stamp": {"git_sha": "0" * 40}, "protocol_hash": "p"}]))
    with pytest.raises(ra.MixedRun):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path)])


def test_the_canonical_directory_needs_a_clean_tree(monkeypatch):
    monkeypatch.setattr(ra, "stamp", lambda: {"git_sha": "a" * 40, "git_dirty": True})
    with pytest.raises(ra.DirtyTree):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2"])


def test_a_trial_run_writes_one_purged_record(monkeypatch, tmp_path):
    _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path / "v2"),
                       "--pred-root", str(tmp_path / "pred")])
    (rec,) = json.loads((tmp_path / "v2" / "metrics_splits.json").read_text())
    assert rec["key"] == "splits.json/cds/family5/aa2"
    assert rec["C"] == rec["C_sweep"][0]["C"] or "C_sweep" in rec       # legacy key names
    from linear_trainer.selection import val_score
    assert val_score(rec) == max(r["macro_f1"] for r in rec["C_sweep"] if r["converged"])
    assert rec["purge"]["rules"] == ["protein@0.40"] and rec["n_test_scored"] == 487 - 28
    assert rec["features"]["featurizer"] == "aa2"
    # A second run resumes: nothing new is written.
    _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path / "v2"),
                       "--pred-root", str(tmp_path / "pred")])
    assert len(json.loads((tmp_path / "v2" / "metrics_splits.json").read_text())) == 1


def test_g1_catches_a_selection_that_reads_test_labels(monkeypatch, tmp_path):
    real_load = sources.load

    def leaky_run_cell(source, task, split, protocol, **kw):
        # A selection that peeks at test labels: its pick moves when they are permuted.
        _, y, _ = sources.load(source, task, "test", split)
        _, y_real, _ = real_load(source, task, "test", split)
        return {"hp": 1.0 if (y == y_real).all() else 2.0, "sweep": []}

    monkeypatch.setattr(ra, "run_cell", leaky_run_cell)
    with pytest.raises(RuntimeError, match="G1"):
        ra.black_box_g1([ra.Cell("splits.json", "cds", "family5", "aa2")], tmp_path)
    assert sources.load is real_load                                         # restored after the check


def test_records_fitted_on_other_features_are_refused(monkeypatch, tmp_path):
    from linear_trainer.protocol import V2
    here = ra.stamp()["git_sha"]
    out = tmp_path / "null_splits_tss_disjoint.json"          # a file this --only run doesn't touch
    out.write_text(json.dumps([{"key": "k", "feature_source": "aa2", "stamp": {"git_sha": here},
                                "protocol_hash": V2.hash, "features": {"sha256": "stale"}}]))
    with pytest.raises(ra.MixedRun, match="other features"):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path)])


def test_a_tree_git_cannot_read_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "stamp", lambda: {"git_sha": None, "git_dirty": True})
    with pytest.raises(ra.DirtyTree):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path)])


def test_a_trial_never_writes_canonical_predictions(monkeypatch, tmp_path):
    with pytest.raises(ra.DirtyTree, match="canonical predictions"):
        _run(monkeypatch, ["--only", "splits.json/cds/family5/aa2", "--out-dir", str(tmp_path),
                           "--pred-root", str(ra.PRED_ROOT)])


@pytest.mark.slow
def test_g1_runs_on_a_real_regression_cell(tmp_path):
    # The Oct 1 review: the permuted run's GenePT targets must not collide with the base run's.
    ra.black_box_g1([ra.Cell("splits.json", "cds", "genept", "aa3")], tmp_path)


def _rec(key, sha="a" * 40, **over):
    return {"key": key, "stamp": {"git_sha": sha}, "protocol_hash": "p", "pad": "x" * 20000, **over}


def _append_many(out, start, n):
    for k in range(start, start + n):
        ra._append(out, _rec(f"k{k}"))


def test_concurrent_appends_keep_every_record(tmp_path):
    """Shards are separate processes sharing one records file: none may lose records."""
    import multiprocessing as mp
    out = tmp_path / "metrics_x.json"
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_append_many, args=(out, 30 * i, 30)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs)
    assert sorted(r["key"] for r in json.loads(out.read_text())) == sorted(f"k{k}" for k in range(120))


def test_a_key_is_never_appended_twice(tmp_path):
    out = tmp_path / "metrics_x.json"
    ra._append(out, _rec("a"))
    with pytest.raises(RuntimeError, match="already recorded"):
        ra._append(out, _rec("a"))


def test_an_append_from_another_commit_is_refused(tmp_path):
    out = tmp_path / "metrics_x.json"
    ra._append(out, _rec("a"))
    with pytest.raises(ra.MixedRun):
        ra._append(out, _rec("b", sha="b" * 40))
    with pytest.raises(ra.MixedRun):
        ra._append(out, _rec("c", protocol_hash="q"))


def test_shards_partition_the_manifest():
    cells = ra.manifest("null")
    parts = [ra.shard(cells, f"{i}/7") for i in range(7)]
    keys = [c.key for part in parts for c in part]
    assert sorted(keys) == sorted(c.key for c in cells) and len(set(keys)) == len(keys)
    with pytest.raises(ValueError):
        ra.shard(cells, "7/7")


def test_only_a_whole_manifest_run_vouches_for_the_records(tmp_path):
    from linear_trainer import records as R
    stamp = {"git_sha": "a" * 40, "protocol_hash": "p"}
    with pytest.raises(R.IncompleteRun, match="no run_complete"):
        R.check_complete(stamp, root=tmp_path)
    cells = [ra.Cell("splits.json", "cds", "family5", "kmer")]
    ra._append(cells[0].out(tmp_path), _rec(cells[0].key))
    ra.write_complete(tmp_path, cells, [{"key": cells[0].key, "hp": 1.0}])
    assert R.check_complete(stamp, root=tmp_path)["n_cells"] == 1
    with pytest.raises(R.IncompleteRun, match="another run"):
        R.check_complete({**stamp, "git_sha": "b" * 40}, root=tmp_path)


def test_an_output_built_from_rewritten_records_is_refused(tmp_path):
    from linear_trainer import records as R
    f = tmp_path / "metrics_splits.json"
    f.write_text("[1]")
    built = {"inputs": R.input_digests(tmp_path)}
    R.check_inputs(built, root=tmp_path)
    f.write_text("[2]")
    with pytest.raises(R.MixedRecords, match="metrics_splits.json"):
        R.check_inputs(built, root=tmp_path)
