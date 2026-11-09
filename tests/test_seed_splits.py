"""G15: the seed splits are files of their own, rebuilt from tracked inputs.

Seed 42 through the seed-split code must reproduce both primary splits exactly;
otherwise the seeds would differ from the primary in more than the seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import make_seed_splits as mss
import make_tss_disjoint_split as mtsd

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = ("train", "val", "test")


def test_seed_42_reproduces_the_primary_cds_split():
    want = json.loads((DATA / "splits.json").read_text())
    got = mss.cds_payload(42)
    for s in SPLITS:
        assert got[s] == want[s]


def test_seed_42_reproduces_the_primary_tss_split():
    want = json.loads((DATA / "splits_tss_disjoint.json").read_text())
    got, _, _ = mtsd.build(mtsd.MANIFEST, mtsd.CLUSTER_TSV, mtsd.FAMILIES, mss.UNIVERSE, 42,
                           "splits_tss_disjoint.json")
    assert got == want


@pytest.mark.parametrize("seed", mss.SEEDS)
def test_tracked_seed_splits_equal_their_rebuild(seed):
    for path, payload in mss.targets(seed).items():
        assert path.read_text() == mss.render(payload), path.name


@pytest.mark.parametrize("seed", mss.SEEDS)
def test_seed_splits_partition_the_primary_genes(seed):
    universe = {g for s in SPLITS for g in json.loads((DATA / "splits.json").read_text())[s]}
    for path in mss.targets(seed):
        parts = json.loads(path.read_text())
        assert sum(len(parts[s]) for s in SPLITS) == len(universe)
        assert set().union(*(set(parts[s]) for s in SPLITS)) == universe


def test_the_primary_seed_is_refused():
    with pytest.raises(ValueError):
        mss.targets(42)


def test_a_reordered_families_parquet_raises(tmp_path):
    df = pd.read_parquet(mss.FAMILIES, columns=["ensembl_id", "family"])
    shuffled = tmp_path / "fams.parquet"
    df.sample(frac=1, random_state=0).to_parquet(shuffled)
    with pytest.raises(mss.RowOrderChanged):
        mss.cds_payload(1, families=shuffled)
