"""G8 and G13: bootstraps rescore stored predictions; chance is a band, not a run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from linear_trainer import stats
from linear_trainer.cell import run_cell
from linear_trainer.protocol import V2
from synth import write_dataset

DATA = Path(__file__).resolve().parents[1] / "data"


def _rec(tmp_path, task="family5", name="a", seed=0, permute=False):
    parquet, splits = write_dataset(tmp_path / name, seed=0, permute_test_labels=False)
    if permute:   # same genes and labels, uninformative features
        import pandas as pd
        df = pd.read_parquet(parquet)
        rng = np.random.default_rng(seed)
        df["x"] = [rng.standard_normal(len(v)).astype(np.float32) for v in df["x"]]
        df.to_parquet(parquet)
    res = run_cell(parquet, task, splits, V2, pred_dir=tmp_path / name / "p")
    return {**res["provenance"], **res["metrics"], "task": task, "split": "toy.json",
            "arm": "cds", "key": f"toy/{task}/{name}"}


def _singletons(rec):
    from linear_trainer.cell import load_predictions
    return {g: g for g in load_predictions(rec)["ids"].tolist()}


def test_holm_matches_a_hand_worked_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    # sorted: d .005*4=.02, a .01*3=.03, c .03*2=.06, b .04*1=.04 -> max(.06, .04)=.06
    assert stats.holm(p) == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert stats.holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}


@pytest.mark.parametrize("task", ["family5", "genept"])
def test_the_bootstrap_point_is_the_record_value(tmp_path, task):
    rec = _rec(tmp_path, task)
    out = stats.cluster_bootstrap(rec, _singletons(rec), n_iters=50)
    # The guard is _scored's exact check; here the interval must bracket the point.
    assert out["point_in_ci"] and out["n_short_class"] == 0


def test_a_record_its_predictions_do_not_reproduce_is_refused(tmp_path):
    rec = _rec(tmp_path)
    rec["test_macro_f1"] += 1e-9
    with pytest.raises(stats.PointMismatch):
        stats.cluster_bootstrap(rec, _singletons(rec), n_iters=5)


def test_grouping_correlated_genes_widens_the_interval(tmp_path):
    # Homologs share their fate: grouping genes whose predictions are all right
    # or all wrong leaves fewer independent units, so the interval must widen.
    from linear_trainer.cell import load_predictions
    rec = _rec(tmp_path)
    single = _singletons(rec)
    a = load_predictions(rec)
    ids = [g for _, g in sorted(zip(a["pred"] == a["y_true"], a["ids"].tolist()))]
    lumped = {g: ids[i // 6] for i, g in enumerate(ids)}          # groups of six, same outcome
    w1 = np.diff(stats.cluster_bootstrap(rec, single, n_iters=400)["ci95"])[0]
    w6 = np.diff(stats.cluster_bootstrap(rec, lumped, n_iters=400)["ci95"])[0]
    assert w6 > w1


def test_a_gene_without_a_group_is_refused(tmp_path):
    rec = _rec(tmp_path)
    groups = _singletons(rec)
    groups.pop(next(iter(groups)))
    with pytest.raises(KeyError):
        stats.cluster_bootstrap(rec, groups, n_iters=5)


def test_a_cell_paired_with_itself_has_no_difference(tmp_path):
    rec = _rec(tmp_path)
    out = stats.paired_bootstrap(rec, rec, _singletons(rec), n_iters=100)
    assert out["delta_point"] == 0 and out["delta_ci95"] == [0.0, 0.0]
    assert out["p_a_gt_b"] == 0 and out["p_one_sided"] == 1.0


def test_a_clearly_better_cell_wins_the_paired_test(tmp_path):
    good = _rec(tmp_path, name="good")
    noise = _rec(tmp_path, name="noise", permute=True)
    noise["splits_sha256"] = good["splits_sha256"]   # same toy split, written twice
    out = stats.paired_bootstrap(good, noise, _singletons(good), n_iters=200)
    assert out["delta_point"] > 0.2
    assert out["p_one_sided"] == pytest.approx(1 / 201) and out["delta_ci95"][0] > 0


def test_cells_on_different_split_files_are_not_paired(tmp_path):
    a, b = _rec(tmp_path, name="a"), _rec(tmp_path, name="b")
    b["splits_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different split files"):
        stats.paired_bootstrap(a, b, _singletons(a), n_iters=5)


@pytest.mark.parametrize("task,field", [("family5", "pred_file"), ("genept", "pred_file"),
                                        ("genept", "targets_file")])
def test_a_tampered_prediction_or_target_file_is_refused(tmp_path, task, field):
    rec = _rec(tmp_path, task)
    path = rec[field] if Path(rec[field]).is_absolute() else Path(__file__).resolve().parents[1] / rec[field]
    arrays = dict(np.load(path, allow_pickle=False))
    key = "pred" if field == "pred_file" else "y_true"
    arrays[key] = arrays[key][::-1].copy()
    np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match="sha256"):
        stats.cluster_bootstrap(rec, n_iters=5, groups={})


def test_the_bootstraps_never_fit(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression, Ridge
    a, b = _rec(tmp_path, name="a"), _rec(tmp_path, name="b", permute=True)

    def refuse(self, *args, **kw):
        raise AssertionError("a bootstrap refitted a probe")
    monkeypatch.setattr(LogisticRegression, "fit", refuse)
    monkeypatch.setattr(Ridge, "fit", refuse)
    stats.cluster_bootstrap(a, _singletons(a), n_iters=5)
    stats.paired_bootstrap(a, b, _singletons(a), n_iters=5)


def _null(n, **over):
    base = {"split": "splits.json", "task": "family5", "feature_source": "kmer",
            "shuffled_labels": True}
    return [{**base, "label_seed": k, "test_macro_f1": 0.15 + k / 1000, "converged": True,
             "edge": False, **over} for k in range(n)]


def test_the_null_band_is_the_central_95_percent():
    band = stats.null_band(_null(200), 200)
    assert band["band95"] == pytest.approx([np.percentile(0.15 + np.arange(200) / 1000, 2.5),
                                            np.percentile(0.15 + np.arange(200) / 1000, 97.5)])


def test_the_null_band_counts_unconverged_refits_and_edges():
    recs = _null(200)
    for r in recs[:3]:
        r["converged"] = False
    recs[5]["edge"] = "nonconverged"
    band = stats.null_band(recs, 200)
    assert band["n_refit_nonconverged"] == 3
    assert band["n_edge"] == {"plateau": 0, "limit": 0, "nonconverged": 1}


def test_a_refit_that_does_not_converge_is_recorded_not_raised(tmp_path, monkeypatch):
    """Decided Oct 1: the cell keeps a non-converged train+val refit and flags it."""
    import linear_trainer.cell as cell_mod
    from linear_trainer.fit import ConvergenceFailure
    real_fit = cell_mod.fit
    calls = []

    def refit_never_converges(kind, X, y, hp, protocol, *, strict=True):
        probe = real_fit(kind, X, y, hp, protocol, strict=False)
        calls.append(strict)
        probe.converged = False
        if strict:
            raise ConvergenceFailure("refit did not converge")
        return probe

    monkeypatch.setattr(cell_mod, "fit", refit_never_converges)
    parquet, splits = write_dataset(tmp_path / "d", seed=0, permute_test_labels=False)
    res = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p")
    assert calls == [False]
    assert res["provenance"]["converged"] is False


def test_a_null_band_must_be_complete_and_single_cell():
    with pytest.raises(ValueError, match="expected 200"):
        stats.null_band(_null(199), 200)
    mixed = _null(199) + _null(1, feature_source="aa2")
    with pytest.raises(ValueError, match="one cell"):
        stats.null_band(mixed, 200)
    with pytest.raises(ValueError, match="shuffled"):
        stats.null_band([{**r, "shuffled_labels": False} for r in _null(200)], 200)


def test_groups_follow_the_split_and_the_arm():
    split = json.loads((DATA / "splits_tss_disjoint.json").read_text())
    genes = {g for s in ("train", "val", "test") for g in split[s]}
    prot = stats.group_of("splits.json", "cds")
    disj = stats.group_of("splits_tss_disjoint.json", "cds")
    assert genes <= prot.keys() and genes <= disj.keys()
    assert stats.group_of("splits.json", "tss") is disj                 # TSS cells: windows count anywhere
    assert stats.group_of("splits_tss_disjoint_seed7.json", "cds") is disj
    assert prot is not disj
    # The disjoint groups are the split's own units: each lies inside one partition...
    where = {g: s for s in ("train", "val", "test") for g in split[s]}
    members: dict[str, set[str]] = {}
    for g in genes:
        members.setdefault(disj[g], set()).add(where[g])
    assert all(len(v) == 1 for v in members.values())
    # ...and they are coarser than protein clusters: window overlap joins some.
    joined = {}
    for g in genes:
        joined.setdefault(disj[g], set()).add(prot[g])
    assert any(len(v) > 1 for v in joined.values())
    assert all(len({disj[g] for g in genes if prot[g] == c}) == 1 for c in {prot[g] for g in genes})


def test_paired_cells_must_score_the_same_genes_and_labels(tmp_path):
    a, b = _rec(tmp_path, name="a"), _rec(tmp_path, name="b")
    b["splits_sha256"] = a["splits_sha256"]
    b2 = {**b, "purge": {**b["purge"], "test_masked": ["x"]}}
    with pytest.raises(ValueError, match="different test genes"):
        stats.paired_bootstrap(a, b2, _singletons(a), n_iters=5)
    with pytest.raises(ValueError, match="share a task"):
        stats.paired_bootstrap(a, {**b, "task": "genept"}, _singletons(a), n_iters=5)


def test_resamples_that_drop_a_family_are_counted_and_capped(tmp_path):
    rec = _rec(tmp_path)
    from linear_trainer.cell import load_predictions
    a = load_predictions(rec)
    # One group per family: a resample of 5 groups misses a family most of the time.
    fam_groups = {g: str(y) for g, y in zip(a["ids"].tolist(), a["y_true"].tolist())}
    with pytest.raises(stats.ShortClassResamples):
        stats.cluster_bootstrap(rec, fam_groups, n_iters=50)


# --- records.load and the selection helpers --------------------------------------

def _write(path, recs):
    path.write_text(json.dumps(recs))


def _fake(key="splits.json/cds/family5/aa2", **over):
    from linear_trainer import records as R
    base = {"key": key, "split": "splits.json", "arm": "cds", "task": "family5",
            "feature_source": key.rsplit("/", 1)[1], "shuffled_labels": False,
            "stamp": {"git_sha": "a" * 40}, "protocol_hash": "p",
            "splits_sha256": R._sha(R.REPO_ROOT / "data" / "splits.json"),
            "purge": {"rules": ["protein@0.40"], **R._purge_inputs(),
                      "split_sha256": R._sha(R.REPO_ROOT / "data" / "splits.json")}}
    return {**base, **over}


def test_records_load_refuses_mixed_or_wrong_files(tmp_path):
    from linear_trainer import records as R
    f = tmp_path / "metrics_splits.json"
    _write(f, [_fake(), _fake("splits.json/cds/family5/aa3", stamp={"git_sha": "b" * 40})])
    with pytest.raises(R.MixedRecords, match="mixes"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(), _fake()])
    with pytest.raises(R.MixedRecords, match="twice"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(purge={"rules": []})])
    with pytest.raises(R.MixedRecords, match="policy"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(purge={**_fake()["purge"], "pairs_sha256": "old"})])
    with pytest.raises(R.MixedRecords, match="pairs"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(purge={k: v for k, v in _fake()["purge"].items() if k != "pairs_sha256"})])
    with pytest.raises(R.MixedRecords, match="pairs"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(splits_sha256="old")])
    with pytest.raises(R.MixedRecords, match="another version"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(purge={**_fake()["purge"], "split_sha256": "old"})])
    with pytest.raises(R.MixedRecords, match="another version"):
        R.load("splits.json", root=tmp_path)
    no_split = {k: v for k, v in _fake()["purge"].items() if k != "split_sha256"}
    _write(f, [_fake(purge=no_split)])
    with pytest.raises(R.MixedRecords, match="another version"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(), _fake("splits.json/cds/family5/aa3", protocol_hash="q")])
    with pytest.raises(R.MixedRecords, match="mixes"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake(split="splits_seed1.json")])
    with pytest.raises(R.MixedRecords, match="names split"):
        R.load("splits.json", root=tmp_path)
    _write(f, [_fake()])
    assert list(R.load("splits.json", root=tmp_path)) == ["splits.json/cds/family5/aa2"]
    other = {"k": _fake(stamp={"git_sha": "c" * 40})}
    with pytest.raises(R.MixedRecords, match="stamps"):
        R.stamp_of(R.load("splits.json", root=tmp_path), other)


def test_selection_refuses_a_partial_candidate_set():
    from linear_trainer import records as R
    sweep = [{"C": 1.0, "macro_f1": 0.5, "converged": True}]
    by = {"kmer": {"C_sweep": sweep}}
    with pytest.raises(R.MissingRecord, match="kmer6"):
        R.best_nt_kmer(by)
    by["kmer6"] = {"C_sweep": [{"C": 1.0, "macro_f1": 0.6, "converged": True}]}
    assert R.best_nt_kmer(by) == "kmer6"


@pytest.mark.parametrize("edge,degenerate,converged,ok", [
    (False, False, True, True), ("plateau", False, True, True), ("limit", False, True, False),
    ("nonconverged", False, True, False), (False, True, True, False), (False, False, False, False)])
def test_a_confirmatory_test_refuses_an_edge_or_degenerate_cell(edge, degenerate, converged, ok):
    import build_statistics as bs
    rec = {"key": "x", "edge": edge, "degenerate": degenerate, "converged": converged}
    if ok:
        bs.check_confirmatory_cell(rec)
    else:
        with pytest.raises(bs.UnsoundConfirmatoryCell):
            bs.check_confirmatory_cell(rec)


def test_the_confirmatory_family_refuses_an_unsound_cell(monkeypatch):
    """confirmatory() itself runs the guard, on both sides of every test."""
    import build_statistics as bs
    from linear_trainer import records as R
    monkeypatch.setattr(bs.stats, "paired_bootstrap", lambda a, b, n_iters: {"p_one_sided": 0.01})
    monkeypatch.setattr(R, "best_encoder", lambda by, arm: "nt_v2_meanD")
    monkeypatch.setattr(R, "best_nt_kmer", lambda by: "kmer")
    monkeypatch.setattr(R, "best_aa", lambda by: "aa2")
    monkeypatch.setattr(R, "best_pool", lambda by, e, arm: f"{'tss_' if arm == 'tss' else ''}{e}_meanD")

    def recs(bad=None):
        out = {}
        for arm, src in [("cds", "nt_v2_meanD"), ("cds", "kmer"), ("cds", "aa2"),
                         ("cds", "esm2_650m"), ("tss", "tss_nt_v2_meanD")]:
            rec = {"key": src, "arm": arm, "task": "family5", "feature_source": src,
                   "shuffled_labels": False, "edge": False, "degenerate": False, "converged": True}
            if src == bad:
                rec["edge"] = "limit"
            out[src] = rec
        return out

    assert set(bs.confirmatory(recs(), recs(), 10)) == {
        "T1 encoder > nucleotide k-mer", "T2 encoder > amino-acid k-mer",
        "T3 ESM-2 650M > encoder", "T4 CDS > TSS (same encoder)"}
    for bad in ("kmer", "esm2_650m"):
        with pytest.raises(bs.UnsoundConfirmatoryCell):
            bs.confirmatory(recs(bad), recs(), 10)
    with pytest.raises(bs.UnsoundConfirmatoryCell):
        bs.confirmatory(recs(), recs("tss_nt_v2_meanD"), 10)


def test_a_disjoint_record_needs_its_window_manifest_hash(tmp_path):
    from linear_trainer import records as R
    name = "splits_tss_disjoint.json"
    sha = R._sha(R.REPO_ROOT / "data" / name)
    inputs = R._purge_inputs()
    rec = _fake(f"{name}/cds/family5/aa2", split=name, splits_sha256=sha,
                purge={"rules": ["protein@0.40", "window"], "split_sha256": sha,
                       "pairs_sha256": inputs["pairs_sha256"]})
    _write(tmp_path / "metrics_splits_tss_disjoint.json", [rec])
    with pytest.raises(R.MixedRecords, match="windows"):
        R.load(name, root=tmp_path)
    rec["purge"]["windows_sha256"] = inputs["windows_sha256"]
    _write(tmp_path / "metrics_splits_tss_disjoint.json", [rec])
    assert list(R.load(name, root=tmp_path)) == [rec["key"]]
