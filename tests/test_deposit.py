"""G30: the Zenodo deposit holds the files the canonical records read, at the hashes
they stamp, keeps NT-v2-derived files apart under CC BY-NC-SA, and packs the same
bytes every time."""
from __future__ import annotations

import hashlib
import tarfile

import numpy as np
import pytest

import build_deposit as bd
from linear_trainer.cell import arrays_sha256


def _pred(root, rel, arrays):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {"pred_file": rel, "pred_sha256": arrays_sha256(arrays)}


def _features(root, rel, data=b"parquet bytes"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"features": {"path": rel, "sha256": hashlib.sha256(data).hexdigest()}}


def _record(root, source, n=3):
    arrays = {"ids": np.array([f"g{i}" for i in range(n)]), "pred": np.arange(n)}
    return {"feature_source": source,
            **_pred(root, f"outputs/predictions/v2/splits/{source}__family5__x.npz", arrays),
            **_features(root, f"data/dataset_{source}.parquet")}


def test_stamped_collects_predictions_and_features(tmp_path):
    rec = _record(tmp_path, "dnabert2_meanD")
    stamps = bd.stamped([rec])
    assert stamps[rec["pred_file"]].kind == "arrays"
    assert stamps[rec["features"]["path"]].kind == "file"
    assert {s.source for s in stamps.values()} == {"dnabert2_meanD"}


def test_featurizer_inputs_are_stamped_with_their_own_licence(tmp_path):
    rec = {"feature_source": "gc", **_pred(tmp_path, "outputs/predictions/v2/splits/gc.npz",
                                           {"pred": np.arange(2)}),
           "features": {"featurizer": "gc", "cds_manifest": "data/cds_manifest.tsv",
                        "cds_manifest_sha256": "a" * 64,
                        "meta": "data/dataset_nt_v2_meanmean.parquet", "meta_sha256": "b" * 64}}
    stamps = bd.stamped([rec])
    assert stamps["data/cds_manifest.tsv"].source == "gc"
    assert bd.licence_of_source(stamps["data/dataset_nt_v2_meanmean.parquet"].source) == bd.NC_LICENCE


def test_verify_passes_on_intact_files(tmp_path):
    bd.verify_stamped(tmp_path, bd.stamped([_record(tmp_path, "dnabert2_meanD")]))


def test_verify_refuses_a_rewritten_prediction(tmp_path):
    rec = _record(tmp_path, "dnabert2_meanD")
    np.savez(tmp_path / rec["pred_file"], ids=np.array(["g0", "g1", "g2"]), pred=np.array([2, 1, 0]))
    with pytest.raises(bd.DepositError, match="does not match"):
        bd.verify_stamped(tmp_path, bd.stamped([rec]))


def test_verify_refuses_a_stale_feature_file(tmp_path):
    rec = _record(tmp_path, "tss_dnabert2_tssanchored")
    (tmp_path / rec["features"]["path"]).write_bytes(b"May-era parquet")
    with pytest.raises(bd.DepositError, match="does not match"):
        bd.verify_stamped(tmp_path, bd.stamped([rec]))


def test_one_path_with_two_stamps_is_refused(tmp_path):
    a, b = _record(tmp_path, "dnabert2_meanD"), _record(tmp_path, "dnabert2_meanD")
    b["features"]["sha256"] = "0" * 64
    with pytest.raises(bd.DepositError, match="two hashes"):
        bd.stamped([a, b])


def test_exact_files_refuses_extras_and_gaps(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    for name in ("A.npz", "B.npz"):
        (d / name).write_bytes(b"x")
    assert bd.exact_files(d, ["A", "B"], ".npz") == [d / "A.npz", d / "B.npz"]
    with pytest.raises(bd.DepositError, match="unexpected"):
        bd.exact_files(d, ["A"], ".npz")
    with pytest.raises(bd.DepositError, match="missing"):
        bd.exact_files(d, ["A", "B", "C"], ".npz")


@pytest.mark.parametrize("source, nc", [
    ("nt_v2_meanD", True),
    ("tss_nt_v2_meanG", True),
    ("tss_nt_v2_tssanchored", True),
    ("tss_nt_v2_centermean", True),
    # Composition of the DNA under NT-v2's chunks: no NT-v2 output in it.
    ("tss_nt_v2_chunk4mergc", False),
    ("tss_nt_v2_chunk6mer", False),
    ("dnabert2_meanD", False),
    ("tss_gena_lm_tssanchored", False),
    ("aa2", False),
    ("kmer6_len", False),
    ("enformer_trunk_global", False),
    ("esm2_650m", False),
])
def test_nt_v2_outputs_are_non_commercial(source, nc):
    assert (bd.licence_of_source(source) == bd.NC_LICENCE) is nc


def test_parts_keep_nc_files_out_of_cc_by_parts(tmp_path):
    recs = [_record(tmp_path, s) for s in ("nt_v2_meanD", "dnabert2_meanD")]
    parts = bd.prediction_parts(bd.stamped(recs))
    by_licence = {p.licence: p for p in parts}
    assert [str(f) for f in by_licence[bd.NC_LICENCE].files] == [recs[0]["pred_file"]]
    assert [str(f) for f in by_licence[bd.CC_BY].files] == [recs[1]["pred_file"]]


@pytest.mark.parametrize("rel", ["docs/reviews/mlcb2026/raw_reviews.md", "STATUS.md", "MINA.md",
                                 "logs/run.log", "data/../STATUS.md"])
def test_pack_refuses_private_paths(tmp_path, rel):
    with pytest.raises(bd.DepositError, match="private"):
        bd.check_public(rel)


def test_pack_is_byte_identical_across_runs(tmp_path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "b.txt").write_text("bee")
    (root / "data" / "a.txt").write_text("ay")
    part = bd.Part("t.tar.gz", bd.CC_BY, ("data/b.txt", "data/a.txt"), "test")
    one = bd.pack_part(root, part, tmp_path / "out1")
    (root / "data" / "a.txt").touch()   # a new mtime must not change the archive
    two = bd.pack_part(root, part, tmp_path / "out2")
    assert one["sha256"] == two["sha256"]
    with tarfile.open(tmp_path / "out1" / "t.tar.gz") as tar:
        assert tar.getnames() == ["data/a.txt", "data/b.txt"]
        assert {m.mtime for m in tar.getmembers()} == {0}


def test_sha256sums_lists_every_part(tmp_path):
    entries = [{"name": "b.tar.gz", "sha256": "2" * 64}, {"name": "a.tar.gz", "sha256": "1" * 64}]
    bd.write_sha256sums(tmp_path, entries)
    assert (tmp_path / "SHA256SUMS").read_text() == f"{'1' * 64}  a.tar.gz\n{'2' * 64}  b.tar.gz\n"


def test_real_records_stamp_files_that_match():
    """The canonical records (tag camera-ready-recompute-v1) against the files on disk."""
    if not (bd.REPO_ROOT / "outputs" / "predictions" / "v2").exists():
        pytest.skip("no stored predictions on this machine")
    stamps = bd.stamped(bd.load_records())
    bd.verify_stamped(bd.REPO_ROOT, {k: v for k, v in stamps.items() if v.kind == "file"})
    assert all(not k.startswith("/") and ".." not in k for k in stamps)
