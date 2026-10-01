"""G2: the evaluation purge masks leaky val and test genes, and only those."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linear_trainer.cell import load_predictions, run_cell, scored_predictions
from linear_trainer.fit import score
from linear_trainer.protocol import V2
from splits.leaks import NONE, Purge, leaky, purge_for, rules_for, window_pairs
from synth import sha256_file, write_dataset

DATA = Path(__file__).resolve().parents[1] / "data"

# measurements_2026-09.md §1: Rule-A leaky test genes on data/splits.json (Sept 25).
SEPT25_TEST = set(
    "ACKR2 CCR6 CCR7 FCRL2 HES2 HOXD9 HTR1A ID3 KCNJ13 KCNJ15 MRGPRE MYOD1 NRL OR1P1 OR4A8 "
    "OVOL3 STK10 TRPV4 ZBTB42 ZKSCAN5 ZNF213 ZNF322 ZNF35 ZNF500 ZNF584 ZNF664 ZNF774 ZNF785".split())


# --- the rule ----------------------------------------------------------------

def test_only_later_splits_are_masked():
    split = {"train": ["t1", "t2"], "val": ["v1", "v2"], "test": ["x1", "x2"]}
    pairs = [("t1", "v1"),   # val with a train partner: masked
             ("v2", "x1"),   # test with a val partner: masked (refit sees val); val side is not
             ("t2", "x2"),   # test with a train partner: masked
             ("t1", "t2"),   # same split: nothing
             ("x1", "x2"),
             ("t1", "zz")]   # a gene outside the split file: nothing
    val, test = leaky(split, pairs)
    assert val == {"v1"} and test == {"x1", "x2"}


def test_every_split_the_recompute_uses_has_a_rule():
    assert rules_for("splits.json", "cds") == ("protein@0.40",)
    assert rules_for("splits.json", "tss") == ("protein@0.40",)          # windows unmasked: the arm's point
    assert rules_for("splits_seed7.json", "cds") == ("protein@0.40",)
    assert rules_for("splits_homology70.json", "cds") == ("protein@0.70",)
    assert rules_for("splits_tss_disjoint.json", "tss") == ("protein@0.40", "window")
    assert rules_for("splits_tss_disjoint_seed1.json", "cds") == ("protein@0.40", "window")
    assert rules_for("splits_random.json", "cds") == ()                 # the leakage demo
    with pytest.raises(KeyError):
        rules_for("splits_new.json", "cds")
    with pytest.raises(ValueError):
        rules_for("splits.json", "enformer")


# --- the tracked tables reproduce the Sept 25 measurement --------------------

def test_the_purge_reproduces_the_sept25_leak_list():
    sym = dict(pd.read_parquet(DATA / "dataset_esm2_650m.parquet",
                               columns=["ensembl_id", "symbol"]).values)
    p = purge_for(DATA / "splits.json", "cds")
    assert {sym[g] for g in p.test} == SEPT25_TEST
    assert len(p.val) == 25
    assert p.stamp["split_sha256"] == sha256_file(DATA / "splits.json")


@pytest.mark.parametrize("name", ["splits_tss_disjoint.json", "splits_tss_disjoint_seed1.json",
                                  "splits_tss_disjoint_seed7.json", "splits_tss_disjoint_seed123.json"])
def test_disjoint_splits_have_no_window_leaks(name):
    assert leaky(json.loads((DATA / name).read_text()), window_pairs()) == (set(), set())


def test_the_random_split_is_not_purged():
    assert purge_for(DATA / "splits_random.json", "cds") is NONE


# --- run_cell applies it -----------------------------------------------------

def _toy(tmp_path):
    parquet, splits = write_dataset(tmp_path)
    s = json.loads(splits.read_text())
    stamp = {"split_sha256": sha256_file(splits)}
    purge = Purge(rules=("protein@0.40",), val=frozenset(s["val"][:7]),
                  test=frozenset(s["test"][:9]), stamp=stamp)
    return parquet, splits, s, purge


def test_purged_test_genes_are_not_scored(tmp_path):
    parquet, splits, s, purge = _toy(tmp_path)
    res = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    rec = {**res["provenance"], **res["metrics"]}
    full, kept = load_predictions(rec), scored_predictions(rec)
    assert len(full["ids"]) == len(s["test"])                       # every prediction is stored
    assert set(kept["ids"]) == set(s["test"]) - purge.test         # only unmasked genes are scored
    assert res["metrics"]["test_macro_f1"] == score("logistic", kept["y_true"], kept["pred"])["test_macro_f1"]
    assert res["metrics"]["unpurged"] == score("logistic", full["y_true"], full["pred"])
    assert res["metrics"]["n_test_scored"] == len(s["test"]) - 9
    assert rec["purge"]["test_masked"] == sorted(purge.test)


def test_purged_val_genes_are_not_used_for_selection(tmp_path, monkeypatch):
    import linear_trainer.cell as cell
    parquet, splits, s, purge = _toy(tmp_path)
    seen = {}
    real = cell.select

    def spy(kind, X_tr, y_tr, X_va, y_va, *a, **kw):
        seen["n_val"] = len(y_va)
        return real(kind, X_tr, y_tr, X_va, y_va, *a, **kw)

    monkeypatch.setattr(cell, "select", spy)
    run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    assert seen["n_val"] == len(s["val"]) - 7


def test_an_empty_purge_changes_nothing(tmp_path):
    parquet, splits, _, purge = _toy(tmp_path)
    empty = Purge(rules=purge.rules, stamp=purge.stamp)
    a = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "a")
    b = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "b", purge=empty)
    assert a["hp"] == b["hp"] and a["sweep"] == b["sweep"]
    assert {k: v for k, v in b["metrics"].items() if k != "unpurged"} == a["metrics"]
    assert a["provenance"]["pred_sha256"] == b["provenance"]["pred_sha256"]


def test_a_purge_from_another_split_file_is_refused(tmp_path):
    parquet, splits, _, purge = _toy(tmp_path)
    wrong = Purge(rules=purge.rules, val=purge.val, test=purge.test, stamp={"split_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="different split file"):
        run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=wrong)


def test_purged_genes_outside_their_split_are_refused(tmp_path):
    parquet, splits, s, purge = _toy(tmp_path)
    stray = Purge(rules=purge.rules, val=purge.val, test=frozenset({s["train"][0]}), stamp=purge.stamp)
    with pytest.raises(RuntimeError, match="not in the test split"):
        run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=stray)


def test_regression_cells_are_purged_too(tmp_path):
    parquet, splits, s, purge = _toy(tmp_path)
    res = run_cell(parquet, "genept", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    kept = scored_predictions({**res["provenance"]})
    assert len(kept["ids"]) == len(s["test"]) - 9
    assert res["metrics"]["test_r2_macro"] == score("ridge", kept["y_true"], kept["pred"])["test_r2_macro"]


# --- the pair table is reproducible -------------------------------------------

@pytest.mark.slow
def test_the_pair_table_matches_a_rebuild():
    import shutil
    import build_protein_pairs as bpp
    mmseqs = shutil.which("mmseqs") or str(Path.home() / ".local" / "bin" / "mmseqs")
    if not Path(mmseqs).exists():
        pytest.skip("MMseqs2 not installed")
    rows = bpp.rule_a_pairs(bpp.search(bpp.proteins(), mmseqs))
    assert bpp.render(rows) == bpp.OUT.read_text()


def test_the_pair_table_is_rule_a_at_40_percent():
    t = pd.read_csv(DATA / "leaks" / "protein_pairs.tsv", sep="\t")
    assert (t.gene_a < t.gene_b).all() and not t.duplicated(["gene_a", "gene_b"]).any()
    assert (t.fident >= 0.4).all() and (t.qcov >= 0.8).all() and (t.tcov >= 0.8).all()
    assert (t.evalue <= 1e-3).all()
    meta = json.loads((DATA / "leaks" / "protein_pairs.json").read_text())
    assert meta["n_pairs"] == len(t)
    assert meta["table_sha256"] == sha256_file(DATA / "leaks" / "protein_pairs.tsv")


# --- review follow-ups (Oct 1) -------------------------------------------------

def test_the_refit_uses_all_of_train_and_val(tmp_path, monkeypatch):
    import linear_trainer.cell as cell
    parquet, splits, s, purge = _toy(tmp_path)
    rows = {}
    real = cell.fit

    def spy(kind, X, y, *a, **kw):
        rows["n"] = len(y)
        return real(kind, X, y, *a, **kw)

    monkeypatch.setattr(cell, "fit", spy)
    run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    assert rows["n"] == len(s["train"]) + len(s["val"])


def test_selection_sees_exactly_the_unmasked_val_rows(tmp_path, monkeypatch):
    import linear_trainer.cell as cell
    from linear_trainer import sources
    parquet, splits, s, purge = _toy(tmp_path)
    X_va, _, ids_va = sources.load(parquet, "family5", "val", splits)
    want = X_va[~np.isin(ids_va, sorted(purge.val))]
    seen = {}
    real = cell.select

    def spy(kind, X_tr, y_tr, Xv, yv, *a, **kw):
        seen["X"] = Xv
        return real(kind, X_tr, y_tr, Xv, yv, *a, **kw)

    monkeypatch.setattr(cell, "select", spy)
    run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    np.testing.assert_array_equal(seen["X"], want)


def test_a_stray_val_gene_is_refused_before_selection(tmp_path, monkeypatch):
    import linear_trainer.cell as cell
    parquet, splits, s, purge = _toy(tmp_path)
    stray = Purge(rules=purge.rules, val=frozenset({s["train"][0]}), test=purge.test, stamp=purge.stamp)
    monkeypatch.setattr(cell, "select", lambda *a, **k: pytest.fail("selection ran"))
    with pytest.raises(RuntimeError, match="not in the val split"):
        run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=stray)


def test_a_purge_that_empties_a_test_family_is_refused(tmp_path):
    parquet, splits, s, purge = _toy(tmp_path)
    fam = dict(pd.read_parquet(parquet, columns=["ensembl_id", "family"]).values)
    one = frozenset(g for g in s["test"] if fam[g] == fam[s["test"][0]])
    wide = Purge(rules=purge.rules, test=one, stamp=purge.stamp)
    with pytest.raises(RuntimeError, match="purged test"):
        run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=wide)


def test_a_record_without_a_purge_field_is_not_rescored(tmp_path):
    parquet, splits, _, purge = _toy(tmp_path)
    res = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    rec = dict(res["provenance"])
    del rec["purge"]
    with pytest.raises(RuntimeError, match="no purge field"):
        scored_predictions(rec)


def test_masked_genes_missing_from_the_predictions_are_refused(tmp_path):
    parquet, splits, _, purge = _toy(tmp_path)
    res = run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p", purge=purge)
    rec = {**res["provenance"]}
    rec["purge"] = {**rec["purge"], "test_masked": rec["purge"]["test_masked"] + ["ENSG_NOT_THERE"]}
    with pytest.raises(RuntimeError, match="missing from the stored predictions"):
        scored_predictions(rec)


def test_the_70_percent_rule_filters_the_table():
    t = pd.read_csv(DATA / "leaks" / "protein_pairs.tsv", sep="\t")
    split = json.loads((DATA / "splits_homology70.json").read_text())
    hand = leaky(split, list(zip(t.gene_a[t.fident >= 0.7], t.gene_b[t.fident >= 0.7])))
    p = purge_for(DATA / "splits_homology70.json", "cds")
    assert (set(p.val), set(p.test)) == hand
    assert (set(p.val), set(p.test)) != leaky(split, list(zip(t.gene_a, t.gene_b)))


def test_a_split_outside_the_table_universe_is_refused(tmp_path):
    from splits.leaks import UncoveredGenes
    split = json.loads((DATA / "splits.json").read_text())
    split["test"] = split["test"] + ["ENSG_NOT_SEARCHED"]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(split))
    with pytest.raises(UncoveredGenes):
        purge_for(path, "cds")


def test_the_arm_follows_the_source():
    from splits.leaks import arm_of
    assert arm_of("nt_v2_meanD") == "cds" and arm_of("esm2_650m") == "cds" and arm_of("aa3") == "cds"
    assert arm_of("tss_nt_v2_meanD") == "tss" and arm_of("enformer_trunk_global") == "tss"
    assert arm_of(Path("data/dataset_tss_dnabert2_tssanchored.parquet")) == "tss"
    assert arm_of(Path("data/dataset_esm2_650m.parquet")) == "cds"


def test_a_cell_on_a_real_split_is_purged_by_default(tmp_path):
    res = run_cell("aa2", "family5", DATA / "splits.json", V2, pred_dir=tmp_path / "p")
    rec = res["provenance"]
    assert rec["purge"]["rules"] == ["protein@0.40"] and len(rec["purge"]["test_masked"]) == 28
    assert res["metrics"]["n_test_scored"] == 487 - 28
