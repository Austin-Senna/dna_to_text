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


@pytest.mark.parametrize("task", ["family5", "genept"])
def test_an_excluded_subset_is_dropped_from_scoring(tmp_path, task):
    from linear_trainer.cell import load_predictions
    from linear_trainer.fit import score
    rec = _rec(tmp_path, task)
    a = load_predictions(rec)
    drop = frozenset(a["ids"][::4].tolist())
    keep = ~np.isin(a["ids"], sorted(drop))
    key = stats._METRIC[stats._KIND[task]]
    out = stats.cluster_bootstrap(rec, _singletons(rec), n_iters=20, exclude=drop)
    assert out["point"] == score(stats._KIND[task], a["y_true"][keep], a["pred"][keep])[key]
    assert (out["n_test"], out["n_excluded"]) == (int(keep.sum()), len(drop))
    assert out["n_groups"] == int(keep.sum())
    # No exclusion: today's output, the record's own value, and no n_excluded field.
    plain = stats.cluster_bootstrap(rec, _singletons(rec), n_iters=20)
    assert plain == stats.cluster_bootstrap(rec, _singletons(rec), n_iters=20, exclude=frozenset())
    assert plain["point"] == rec[key] and "n_excluded" not in plain


def test_a_paired_exclusion_drops_the_same_genes_from_both_sides(tmp_path, monkeypatch):
    from linear_trainer.cell import load_predictions
    from linear_trainer.fit import score
    a, b = _rec(tmp_path, name="a"), _rec(tmp_path, name="b", permute=True)
    # b's rows come back in another gene order, so a mask built on one side's
    # order and applied to the other's would hit the wrong genes.
    scored = stats._scored
    def shuffled(rec):
        arrays = scored(rec)
        if rec is not b:
            return arrays
        perm = np.random.default_rng(1).permutation(len(arrays["ids"]))
        return {k: v[perm] for k, v in arrays.items()}
    monkeypatch.setattr(stats, "_scored", shuffled)
    ids = load_predictions(a)["ids"]
    drop = frozenset(ids[1::3].tolist())
    out = stats.paired_bootstrap(a, b, _singletons(a), n_iters=20, exclude=drop)
    def f1(rec):
        p = load_predictions(rec)
        keep = ~np.isin(p["ids"], sorted(drop))
        return score("logistic", p["y_true"][keep], p["pred"][keep])["test_macro_f1"]
    assert out["delta_point"] == pytest.approx(f1(a) - f1(b), abs=1e-12)
    assert (out["n_test"], out["n_excluded"]) == (len(ids) - len(drop), len(drop))
    # The exclusion never excuses cells that score different genes.
    b["purge"] = {**b["purge"], "test_masked": [ids[0]]}
    with pytest.raises(ValueError, match="different test genes"):
        stats.paired_bootstrap(a, b, _singletons(a), n_iters=5, exclude=drop)


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
    assert stats.group_of("splits.json", "tss") is stats.group_of("splits_tss_disjoint.json", "cds")
    assert stats.group_of("splits_tss_disjoint_seed7.json", "cds") is stats.group_of("splits.json", "tss")
    assert stats.group_of("splits.json", "cds") is not stats.group_of("splits.json", "tss")
    prot, disj = stats._protein_groups(), stats._disjoint_groups()       # the bases, before G24's joins
    assert genes <= prot.keys() and genes <= disj.keys()
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
    from linear_trainer.protocol import V2
    base = {"key": key, "split": "splits.json", "arm": "cds", "task": "family5",
            "feature_source": key.rsplit("/", 1)[1], "shuffled_labels": False,
            "stamp": {"git_sha": "a" * 40}, "protocol_hash": V2.hash,
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
    # A trial fitted at another thread count (recompute_all --threads) never reaches a builder.
    from dataclasses import replace
    from linear_trainer.protocol import V2
    _write(f, [_fake(protocol_hash=replace(V2, threads=6).hash)])
    with pytest.raises(R.MixedRecords, match="protocol"):
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


FAMILIES = {"gpcr": 82, "immune": 22, "ion": 27, "kinase": 83, "tf": 245}


@pytest.mark.parametrize("edge,degenerate,converged,by_class,ok", [
    (False, False, True, FAMILIES, True), ("plateau", False, True, FAMILIES, True),
    ("limit", False, True, FAMILIES, False), ("nonconverged", False, True, FAMILIES, False),
    (False, True, True, FAMILIES, False), (False, False, False, FAMILIES, False),
    (False, False, True, {**FAMILIES, "immune": 14}, False),             # G28: a thinned family
    (False, False, True, {k: v for k, v in FAMILIES.items() if k != "ion"}, False)])
def test_a_confirmatory_test_refuses_an_edge_or_degenerate_cell(edge, degenerate, converged, by_class, ok):
    import build_statistics as bs
    rec = {"key": "x", "edge": edge, "degenerate": degenerate, "converged": converged,
           "n_test_scored_by_class": by_class}
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
                   "shuffled_labels": False, "edge": False, "degenerate": False, "converged": True,
                   "n_test_scored_by_class": FAMILIES}
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


def test_statistics_need_a_passing_reproduction_of_the_current_records(tmp_path):
    """Rule 3: no reproduction, a failed one, or one of other records files is refused."""
    from linear_trainer import records as R
    _write(tmp_path / "metrics_splits.json", [{"k": 1}])
    verdict = {"ok": True, "inputs": R.input_digests(tmp_path), "failures": [],
               "cells_run": list(range(1, 14)), "cells_expected": list(range(1, 14)), "n_boot": 2000}
    with pytest.raises(R.NotReproduced, match="no reproduction"):
        R.check_reproduced(tmp_path)
    for bad, match in [({"ok": False, "failures": ["cell 1: pred"]}, "failures"),
                       ({"inputs": {}}, "other versions"),
                       ({"cells_run": [5]}, "partial"),                      # --cells 5 passes, vouches for nothing
                       ({"cells_run": list(range(1, 11))}, "partial"),       # the edge-branch cells skipped
                       ({"n_boot": 200}, "partial")]:
        _write(tmp_path / R.REPRODUCED, {**verdict, **bad})
        with pytest.raises(R.NotReproduced, match=match):
            R.check_reproduced(tmp_path)
    _write(tmp_path / R.REPRODUCED, verdict)
    assert R.check_reproduced(tmp_path)["ok"]
    _write(tmp_path / "null_splits.json", [{"k": 2}])     # a records file added since
    with pytest.raises(R.NotReproduced, match="other versions"):
        R.check_reproduced(tmp_path)


@pytest.mark.parametrize("field,value", [(None, None), ("b", "splits.json/cds/family5/kmer"),
                                         ("delta_point", 0.0500001), ("n_groups", 425),
                                         ("delta_ci95", [0.01, 0.12]), ("p_one_sided", 0.11)])
def test_the_confirmatory_tests_must_agree_with_the_reproduction(field, value):
    import build_statistics as bs
    from linear_trainer import records as R
    ours = {"a": "splits.json/cds/family5/nt_v2_meanG", "b": "splits.json/cds/family5/kmer6",
            "delta_point": 0.05, "n_test": 459, "n_groups": 426, "delta_ci95": [-0.02, 0.12],
            "p_one_sided": 0.07}
    theirs = {**ours, "delta_ci95": [-0.015, 0.125], "p_one_sided": 0.08}   # Monte Carlo noise
    if field is None:
        bs.check_reproduction({"T1 encoder > nucleotide k-mer": ours}, {"T1": theirs})
        with pytest.raises(R.NotReproduced, match="no result"):
            bs.check_reproduction({"T2 encoder > amino-acid k-mer": ours}, {"T1": theirs})
        return
    with pytest.raises(R.NotReproduced, match=field):
        bs.check_reproduction({"T1 encoder > nucleotide k-mer": ours}, {"T1": {**theirs, field: value}})


@pytest.mark.parametrize("split,arm", [("splits.json", "cds"), ("splits_tss_disjoint.json", "cds"),
                                       ("splits.json", "tss")])
def test_no_dependency_spans_two_resampling_units(split, arm):
    """G24: a Rule-A pair always shares a unit (MMseqs2 clusters alone split some),
    and for GenePT so do genes with one summary template; family5 units ignore
    templates."""
    from data_loader import label_audit
    from splits.leaks import PAIR_MIN_ID, read_protein_pairs
    gt, _, _ = label_audit.load_inputs()
    templates = [ids for ids in label_audit.shared_summary_groups(gt).values()]
    f5, gp = stats.group_of(split, arm, "family5"), stats.group_of(split, arm, "genept")
    pairs = read_protein_pairs(PAIR_MIN_ID)
    for groups in (f5, gp):
        assert all(groups[a] == groups[b] for a, b in pairs if a in groups and b in groups)
    assert all(len({gp[g] for g in ids}) == 1 for ids in templates)
    assert any(len({f5[g] for g in ids}) > 1 for ids in templates)
    # Every unit is a union of whole base groups.
    base = stats._disjoint_groups() if stats.group_of(split, arm) is stats._groups(True, False) \
        else stats._protein_groups()
    assert all(len({gp[g] for g in base if base[g] == c}) == 1 for c in set(base.values()))


def test_the_unit_check_fails_without_the_joins(monkeypatch):
    """The guard above must be able to fail: the bare protein clusters split a Rule-A pair."""
    from splits.leaks import PAIR_MIN_ID, read_protein_pairs
    base = stats._protein_groups()
    pairs = read_protein_pairs(PAIR_MIN_ID)
    assert not all(base[a] == base[b] for a, b in pairs if a in base and b in base)


def _fake_scored(monkeypatch, ids, y, pred):
    monkeypatch.setattr(stats, "_scored", lambda rec: {"ids": np.asarray(ids), "y_true": y, "pred": pred})


def test_within_family_r2_credits_only_signal_beyond_the_family(monkeypatch):
    rng = np.random.default_rng(0)
    fam = np.repeat(["a", "b", "c"], 40)
    means = {f: rng.normal(size=6) * 5 for f in "abc"}
    off = np.stack([means[f] for f in fam])
    resid = rng.normal(size=(120, 6))
    ids = [f"g{i}" for i in range(120)]
    family = dict(zip(ids, fam))
    rec = {"key": "x", "task": "genept", "split": "toy.json", "arm": "cds", "test_r2_macro": 0.9}
    groups = {g: g for g in ids}
    _fake_scored(monkeypatch, ids, off + resid, off)            # the family mean, nothing more
    only_family = stats.within_family_r2(rec, family, means, groups, n_iters=20)
    _fake_scored(monkeypatch, ids, off + resid, off + resid)    # the within-family signal too
    full = stats.within_family_r2(rec, family, means, groups, n_iters=20)
    assert only_family["point"] <= 0 < full["point"] == pytest.approx(1.0)
    from sklearn.metrics import r2_score
    assert r2_score(off + resid, off) > 0.5                     # plain R^2 rewards the family alone


def test_per_cell_chance_separates_signal_from_a_class_mix(monkeypatch):
    rng = np.random.default_rng(1)
    y = rng.choice(np.array(["gpcr", "ion", "kinase", "tf", "immune"]), size=300, p=[.2, .1, .15, .5, .05])
    rec = {"key": "x", "task": "family5", "split": "toy.json", "arm": "cds"}
    ids = [f"g{i}" for i in range(300)]
    from sklearn.metrics import f1_score
    for pred, informative in ((y.copy(), True), (rng.permutation(y), False)):
        _fake_scored(monkeypatch, ids, y, pred)
        rec["test_macro_f1"] = float(f1_score(y, pred, average="macro"))
        out = stats.permutation_chance(rec, n_iters=200)
        assert (out["p_above_chance"] < 0.01) is informative
        assert (out["chance95"][0] <= out["point"] <= out["chance95"][1]) is not informative
