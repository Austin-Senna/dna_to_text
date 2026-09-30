"""The Phase 3 pilot gate: cache comparison and pilot gene choice.

The pilot trusts the AWS extraction only if these comparisons pass, so each
check here must be able to fail: a perturbed, reshaped, missing or NaN array
fails, and exact mode fails on a one-ulp change.
"""
from __future__ import annotations

import numpy as np
import pytest

import compare_extraction_caches as cc
from data_loader.cache_meta import write_npz


def _write_old(d, gene, arrays):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{gene}.npz", **arrays)


def _write_new(d, gene, arrays):
    d.mkdir(parents=True, exist_ok=True)
    write_npz(d / f"{gene}.npz", arrays, {"encoder": "toy"})


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    return {"mean": rng.standard_normal((3, 8)).astype(np.float32),
            "max": rng.standard_normal((3, 8)).astype(np.float32)}


def _pair(tmp_path, b_arrays=None, genes=("ENSG1", "ENSG2")):
    a, b = tmp_path / "a", tmp_path / "b"
    for i, g in enumerate(genes):
        arr = _arrays(i)
        _write_new(a, g, arr)
        _write_old(b, g, arr if b_arrays is None or g != genes[0] else b_arrays)
    return a, b


def test_identical_caches_pass_in_both_modes_across_formats(tmp_path):
    a, b = _pair(tmp_path)  # new format (meta) vs old format (no meta)
    assert cc.compare_dirs(a, b, exact=True).failures == []
    assert cc.compare_dirs(a, b, max_rel_l2=1e-3, min_cos=0.9999).failures == []


def test_esm_npy_and_npz_emb_compare(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    v = np.arange(1, 11, dtype=np.float32)
    write_npz(a / "ENSG1.npz", {"emb": v}, {"model": "esm"})
    np.save(b / "ENSG1.npy", v)
    assert cc.compare_dirs(a, b, exact=True).failures == []
    np.save(b / "ENSG1.npy", v * 1.1)
    assert cc.compare_dirs(a, b, max_rel_l2=1e-3, min_cos=0.9).failures != []


def test_a_perturbation_fails_past_the_threshold_and_passes_below_it(tmp_path):
    base = _arrays(0)
    small = {k: v * (1 + 1e-6) for k, v in base.items()}
    a, b = _pair(tmp_path, b_arrays=small)
    assert cc.compare_dirs(a, b, max_rel_l2=1e-3, min_cos=0.9999).failures == []
    assert cc.compare_dirs(a, b, exact=True).failures != []  # exact means bit-identical
    big = {k: v + 0.1 * np.random.default_rng(9).standard_normal(v.shape).astype(np.float32)
           for k, v in base.items()}
    a, b = _pair(tmp_path / "big", b_arrays=big)
    assert cc.compare_dirs(a, b, max_rel_l2=1e-3, min_cos=0.9999).failures != []


def test_one_ulp_fails_exact(tmp_path):
    base = _arrays(0)
    bumped = {k: v.copy() for k, v in base.items()}
    bumped["mean"][0, 0] = np.nextafter(bumped["mean"][0, 0], np.float32(np.inf))
    a, b = _pair(tmp_path, b_arrays=bumped)
    assert cc.compare_dirs(a, b, exact=True).failures != []


def test_shape_mismatch_missing_gene_nan_and_missing_key_fail(tmp_path):
    a, b = _pair(tmp_path, b_arrays={"mean": np.zeros((4, 8), np.float32),
                                     "max": np.zeros((3, 8), np.float32)})
    assert any("shape" in f for f in cc.compare_dirs(a, b, max_rel_l2=1.0, min_cos=0.0).failures)

    a, b = _pair(tmp_path / "miss")
    (b / "ENSG2.npz").unlink()
    assert any("missing" in f for f in cc.compare_dirs(a, b, exact=True).failures)
    assert any("missing" in f for f in cc.compare_dirs(a, b, exact=True, genes=["ENSG1", "ENSG3"]).failures)

    nan = _arrays(0)
    nan["max"][1, 2] = np.nan
    a, b = _pair(tmp_path / "nan", b_arrays=nan)
    assert any("nan" in f.lower() for f in cc.compare_dirs(a, b, max_rel_l2=1.0, min_cos=0.0).failures)

    a, b = _pair(tmp_path / "key")
    assert any("key" in f for f in cc.compare_dirs(a, b, exact=True, keys=["mean", "cls"]).failures)


def test_a_dropped_array_fails_unless_the_keys_are_named(tmp_path):
    arr = _arrays(0)
    a, b = tmp_path / "a", tmp_path / "b"
    _write_new(a, "ENSG1", arr)
    _write_old(b, "ENSG1", {**arr, "cls": np.ones((3, 8), np.float32)})  # e.g. new code lost cls
    res = cc.compare_dirs(a, b, exact=True)
    assert any("keys differ" in f for f in res.failures) and res.keys_only_in_b == {"cls"}
    assert cc.compare_dirs(a, b, exact=True, keys=["mean", "max"]).failures == []  # HyenaDNA by design


def test_pilot_candidates_keep_only_genes_with_byte_identical_inputs():
    rows = [
        # gene, strand, pad_up, pad_down, start, end
        ("ENSG1", "1", 0, 0, 100, 200),   # kept
        ("ENSG2", "-1", 0, 0, 100, 200),  # minus strand: window is now reverse-complemented
        ("ENSG3", "1", 5, 0, 100, 200),   # padded
        ("ENSG4", "1", 0, 0, 100, 201),   # span moved
        ("ENSG5", "1", 0, 0, 100, 200),   # kept
        ("ENSG6", "1", 0, 0, 100, 200),   # translation exception
        ("ENSG7", "1", 0, 0, 100, 200),   # no May window
    ]
    old_spans = {g: (100, 200) for g in ("ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5", "ENSG6")}
    assert cc.pilot_candidates(rows, old_spans, exceptions={"ENSG6"}) == ["ENSG1", "ENSG5"]


def test_pilot_pick_is_deterministic_and_family_stratified():
    fams = {f"ENSG{i:03d}": ["tf", "kinase", "ion", "gpcr", "zinc"][i % 5] for i in range(60)}
    fams.update({f"ENSG{i:03d}": "tf" for i in range(60, 102)})
    cands = sorted(fams)
    pick = cc.pick_pilot(cands, fams, n=50, seed=0)
    assert pick == cc.pick_pilot(list(reversed(cands)), fams, n=50, seed=0)
    assert len(pick) == 50 and len(set(pick)) == 50 and set(pick) <= set(cands)
    counts = {f: sum(fams[g] == f for g in pick) for f in set(fams.values())}
    assert counts == {"tf": 10, "kinase": 10, "ion": 10, "gpcr": 10, "zinc": 10}
    short = {g: f for g, f in fams.items() if not (f == "ion" and int(g[4:]) >= 20)}  # 4 ion left
    pick = cc.pick_pilot(sorted(short), short, n=50, seed=0)
    assert len(pick) == 50 and sum(short[g] == "ion" for g in pick) == 4


def _stamped(d, gene, stamp):
    d.mkdir(parents=True, exist_ok=True)
    write_npz(d / f"{gene}.npz", {"mean": np.ones((1, 2))},
              {"encoder": "toy", "device": "cuda", **stamp})


A10G = {"device_name": "NVIDIA A10G", "torch": "2.11.0", "cuda": "13.0"}
GENES = ["ENSG1", "ENSG2", "ENSG3"]


def test_census_needs_exactly_the_genes_from_one_stamped_run(tmp_path):
    for g in GENES:
        _stamped(tmp_path / "ok", g, A10G)
    assert cc.census(tmp_path / "ok", GENES).failures == []
    assert cc.census(tmp_path / "ok", GENES + ["ENSG4"]).failures != []  # a gene is missing
    assert cc.census(tmp_path / "ok", GENES[:2]).failures != []  # an unexpected gene

    for g in GENES[:2]:
        _stamped(tmp_path / "mix", g, A10G)
    _stamped(tmp_path / "mix", "ENSG3", {**A10G, "device_name": "NVIDIA GeForce RTX 5060"})
    assert any("runs mixed" in f for f in cc.census(tmp_path / "mix", GENES).failures)

    for g in GENES:
        _stamped(tmp_path / "bare", g, dict.fromkeys(A10G))
    assert any("unstamped" in f for f in cc.census(tmp_path / "bare", GENES).failures)

    (tmp_path / "old").mkdir()
    np.savez(tmp_path / "old" / "ENSG1.npz", mean=np.ones(2))
    assert cc.census(tmp_path / "old", ["ENSG1"]).failures != []  # no meta record


def test_census_refuses_torn_files_and_non_finite_arrays(tmp_path):
    for g in GENES:
        _stamped(tmp_path / "ok", g, A10G)
    (tmp_path / "ok" / "ENSG4.npz.partial").write_bytes(b"PK")
    assert cc.census(tmp_path / "ok", GENES).failures != []  # a killed write

    for g in GENES:
        _stamped(tmp_path / "nan", g, A10G)
    write_npz(tmp_path / "nan" / "ENSG2.npz", {"mean": np.array([[1.0, np.nan]])},
              {"encoder": "toy", "device": "cuda", **A10G})
    assert any("finite" in f for f in cc.census(tmp_path / "nan", GENES).failures)


def test_old_window_bytes_are_checked_against_the_manifest(tmp_path):
    from data_loader.enformer_windows import sha256_seq

    fa = tmp_path / "ENSG1.fa"
    fa.write_text(">chromosome:GRCh38:1:101:108:1\nacgt\nACGN\n")
    assert cc.old_window_sha(fa) == sha256_seq("ACGTACGN")


def test_signed_zero_is_not_bit_identical(tmp_path):
    base = _arrays(0)
    flipped = {k: v.copy() for k, v in base.items()}
    base["mean"][0, 0], flipped["mean"][0, 0] = 0.0, -0.0
    a, b = tmp_path / "a", tmp_path / "b"
    _write_new(a, "ENSG1", base)
    _write_old(b, "ENSG1", flipped)
    assert cc.compare_dirs(a, b, exact=True).failures != []


def test_the_gate_is_the_exit_status(tmp_path, monkeypatch, capsys):
    import sys

    base = _arrays(0)
    a, b = _pair(tmp_path, b_arrays={k: v + 1 for k, v in base.items()})
    run = lambda *extra: (monkeypatch.setattr(sys, "argv", ["x", "compare", str(a), str(b), *extra]),
                          cc.main())[1]
    assert run("--exact") == 1  # a failed comparison is what extract_box.sh records
    assert run("--exact", "--report-only") == 0 and "MEASURED" in capsys.readouterr().out
    a2, b2 = _pair(tmp_path / "same")
    monkeypatch.setattr(sys, "argv", ["x", "compare", str(a2), str(b2), "--exact"])
    assert cc.main() == 0


def test_sha256sums_are_what_sha256sum_check_reads(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "data").mkdir()
    content = b">x\nACGT\n"
    (tmp_path / "data" / "G1.fa").write_bytes(content)
    monkeypatch.setattr(cc, "REPO_ROOT", tmp_path)
    cc._write_sha256sums(tmp_path / "s.sha256", ["data/G1.fa"])
    digest = hashlib.sha256(content).hexdigest()
    assert (tmp_path / "s.sha256").read_text() == f"{digest}  data/G1.fa\n"  # two spaces, relative
