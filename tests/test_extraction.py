"""Guards for feature extraction before the GPU re-extraction (G3, G19, G20, 1C).

HyenaDNA sees DNA only: a boundary token it was never trained with sits in the
causal receptive field of every position and made Mean-CLS a constant vector.
Boundary tokens now come from the encoder spec, not from whatever the tokenizer
declares. Caches refuse files built differently, model loads refuse missing
weights, every model load pins a revision, and a constant feature matrix never
reaches a probe.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class _Tok:
    """Declares CLS/SEP like HyenaDNA's tokenizer, whose own inputs never use CLS."""
    cls_token_id, bos_token_id, sep_token_id, eos_token_id = 0, None, 1, None

    def __call__(self, seq, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [7 + "ACGT".index(b) for b in seq]}


class _CausalModel:
    """Records its inputs; position t sees tokens 0..t, like HyenaDNA."""

    def __init__(self):
        self.seen = []

    def __call__(self, input_ids, attention_mask=None):
        import torch

        self.seen.append(input_ids[0].tolist())
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return type("Out", (), {"last_hidden_state": torch.cumsum(x, dim=1)})


# --- G3: HyenaDNA gets DNA only ----------------------------------------------

def test_boundary_tokens_come_from_the_spec_not_the_tokenizer():
    from data_loader.multi_pool import embed_sequence_multi_pool

    model = _CausalModel()
    red = embed_sequence_multi_pool("ACGTAC", model, _Tok(), "cpu", max_content_tokens=4,
                                    stride=1, boundary_tokens=False)
    assert model.seen == [[7, 8, 9, 10], [10, 7, 8]]  # content only; the last chunk is short
    assert set(red) == {"mean", "max"}

    model = _CausalModel()
    red = embed_sequence_multi_pool("ACGTAC", model, _Tok(), "cpu", max_content_tokens=4,
                                    stride=1, boundary_tokens=True)
    assert model.seen[0] == [0, 7, 8, 9, 10, 1]
    assert set(red) == {"mean", "special_mean", "max", "cls"}


def test_hyena_is_dna_only_with_four_pools_and_22_cds_configs():
    from data_loader.model_registry import ENCODER_SPECS, encoder_pools

    assert ENCODER_SPECS["hyena_dna"].boundary_tokens is False
    assert all(s.boundary_tokens for n, s in ENCODER_SPECS.items() if n != "hyena_dna")
    assert encoder_pools("hyena_dna") == ("meanmean", "maxmean", "meanD", "meanG")
    assert sum(len(encoder_pools(e)) for e in ENCODER_SPECS) == 22
    # the TSS grid never had specialmean: 5 pools, 4 for HyenaDNA
    assert encoder_pools("dnabert2", "TSS") == ("meanmean", "maxmean", "clsmean", "meanD", "meanG")
    assert encoder_pools("hyena_dna", "TSS") == ("meanmean", "maxmean", "meanD", "meanG")


def test_dropped_pools_are_not_registered_or_candidates():
    from linear_trainer.selection import encoder_cells
    from linear_trainer.sources import DATASET_PATHS

    for gone in ("hyena_dna_clsmean", "hyena_dna_specialmean", "tss_hyena_dna_clsmean",
                 "tss_dnabert2_specialmean"):
        assert gone not in DATASET_PATHS, gone
    assert "tss_hyena_dna_centermean" in DATASET_PATHS
    cells = encoder_cells("hyena_dna")
    assert "hyena_dna_clsmean" not in cells and "hyena_dna_specialmean" not in cells
    assert "hyena_dna_meanG" in cells


def _cls(src, val):
    return {"feature_source": src, "task": "family5", "C": 1.0, "test_macro_f1": 0.5,
            "C_sweep": [{"C": 1.0, "macro_f1": val}]}


def test_a_hyena_clsmean_decoy_never_wins_a_pick(monkeypatch):
    import build_paper_tables as bpt
    import headline_cells as hc

    recs = {f"hyena_dna_{p}": _cls(f"hyena_dna_{p}", 0.4) for p in ("meanmean", "meanG")}
    recs["hyena_dna_clsmean"] = _cls("hyena_dna_clsmean", 0.99)
    assert hc.best(recs, "hyena_dna_") != "hyena_dna_clsmean"
    for src, rec in recs.items():
        monkeypatch.setitem(bpt.CLS, src, rec)
    assert bpt.cls_best_pool("hyena_dna")[0] != "clsmean"


# --- G19: extraction caches know what built them ---------------------------------

def _spec(**kw):
    from data_loader.model_registry import EncoderSpec

    base = dict(name="toy", display_name="Toy", model_name="org/toy", model_kind="x",
                cache_name="toy", dataset_stem="toy", loader_module="x",
                max_content_tokens=4, stride=1, revision="abc123", boundary_tokens=False)
    return EncoderSpec(**{**base, **kw})


def test_extraction_cache_refuses_files_built_differently(tmp_path):
    from data_loader.cache_meta import StaleCache
    from data_loader.multi_pool import embed_all_multi_pool

    def load(device):
        return _CausalModel(), _Tok(), "cpu"

    seqs = {"ENSG1": "ACGTAC"}
    first = embed_all_multi_pool(seqs, load, tmp_path, _spec())
    assert set(first["ENSG1"]) == {"mean", "max"}

    def must_not_load(device):
        raise AssertionError("a matching cache must not reload the model")

    again = embed_all_multi_pool(seqs, must_not_load, tmp_path, _spec())
    np.testing.assert_array_equal(again["ENSG1"]["mean"], first["ENSG1"]["mean"])
    with pytest.raises(StaleCache):  # another revision
        embed_all_multi_pool(seqs, must_not_load, tmp_path, _spec(revision="def456"))
    with pytest.raises(StaleCache):  # another input under the same gene id
        embed_all_multi_pool({"ENSG1": "ACGTAA"}, must_not_load, tmp_path, _spec())
    np.savez(tmp_path / "ENSG2.npz", mean=np.zeros((1, 2)), special_mean=np.zeros((1, 2)))
    with pytest.raises(StaleCache):  # a pre-G19 file with no meta
        embed_all_multi_pool({"ENSG2": "ACGT"}, must_not_load, tmp_path, _spec())


# --- model loads: no missing weights, pinned revisions (G20) -------------------

def test_load_checks_refuse_missing_and_unexpected_weights():
    import torch

    from data_loader.load_checks import LoadError, check_loading_info, check_state_dict_load

    lin = torch.nn.Linear(2, 2)
    sd = {"weight": torch.zeros(2, 2)}
    with pytest.raises(LoadError):
        check_state_dict_load(lin.load_state_dict(sd, strict=False), what="toy")
    sd = {**lin.state_dict(), "lm_head.bias": torch.zeros(1)}
    res = lin.load_state_dict(sd, strict=False)
    with pytest.raises(LoadError):
        check_state_dict_load(res, what="toy")
    check_state_dict_load(res, what="toy", allowed_unexpected=("lm_head.",))
    with pytest.raises(LoadError):
        check_loading_info({"missing_keys": ["encoder.x"], "unexpected_keys": [],
                            "mismatched_keys": [], "error_msgs": []}, what="toy")


def test_nt_v2_loader_refuses_a_checkpoint_with_missing_weights(monkeypatch, tmp_path):
    import torch

    import data_loader.nt_v2_encoder as nt
    from data_loader.load_checks import LoadError

    class Masked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.esm = torch.nn.Linear(2, 2)

    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    monkeypatch.setattr(nt, "snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(nt.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(nt.AutoConfig, "from_pretrained", lambda *a, **k: type("C", (), {})())
    monkeypatch.setattr(nt.AutoModelForMaskedLM, "from_config", lambda *a, **k: Masked())
    monkeypatch.setattr(nt.torch, "load", lambda *a, **k: {"esm.weight": torch.zeros(2, 2)})
    with pytest.raises(LoadError):
        nt.load_model("cpu")


def test_every_model_load_pins_a_revision():
    """Static scan: hub loads name a revision; loads from a pinned snapshot dir are fine."""
    unpinned = []
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        text = path.read_text()
        for m in re.finditer(r"(?<!`)\b(from_pretrained|snapshot_download)\(", text):  # not ``docs``
            depth, i = 1, m.end()
            while depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            args = text[m.end():i - 1]
            if "revision=" in args or args.strip().startswith("snapshot_dir"):
                continue
            unpinned.append(f"{path.relative_to(ROOT)}: {m.group(1)}({args.strip()[:50]})")
    assert unpinned == []


# --- G3: constant features never reach a probe ------------------------------------

def test_near_constant_features_are_refused():
    from linear_trainer.sources import ConstantFeatures, check_not_constant
    from splits import load_split

    X, _, _ = load_split("train", dataset_path=DATA / "dataset_hyena_dna_clsmean.parquet")
    with pytest.raises(ConstantFeatures):
        check_not_constant(X, "hyena_dna_clsmean")
    X, _, _ = load_split("train", dataset_path=DATA / "dataset_hyena_dna_meanmean.parquet")
    check_not_constant(X, "hyena_dna_meanmean")


def test_load_refuses_a_constant_source(tmp_path, monkeypatch):
    from linear_trainer import sources
    from synth import write_dataset

    parquet, splits = write_dataset(tmp_path)
    df = pd.read_parquet(parquet)
    df["x"] = [np.full(16, 3.0, dtype=np.float32) + 1e-9 * i for i in range(len(df))]
    df.to_parquet(parquet)
    monkeypatch.setitem(sources.DATASET_PATHS, "toy_const", parquet)
    with pytest.raises(sources.ConstantFeatures):
        sources.load("toy_const", "family5", "train", splits)


def test_single_class_predictions_are_flagged(tmp_path, monkeypatch):
    from linear_trainer import cell
    from linear_trainer.protocol import V2
    from synth import write_dataset

    parquet, splits = write_dataset(tmp_path)
    rec = cell.run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "p")
    assert rec["provenance"]["degenerate"] is False

    real_fit = cell.fit

    def majority_fit(*a, **k):
        probe = real_fit(*a, **k)
        probe.predict = lambda X: np.full(len(X), "tf", dtype=object)
        return probe

    monkeypatch.setattr(cell, "fit", majority_fit)
    rec = cell.run_cell(parquet, "family5", splits, V2, pred_dir=tmp_path / "q")
    assert rec["provenance"]["degenerate"] is True


@pytest.mark.slow
def test_every_registered_source_on_disk_is_not_constant():
    from linear_trainer.sources import DATASET_PATHS, check_not_constant
    from splits import load_split

    checked = 0
    for name, path in sorted(DATASET_PATHS.items()):
        if not Path(path).exists():
            continue
        X, _, _ = load_split("train", dataset_path=Path(path))
        check_not_constant(X, name)
        checked += 1
    assert checked >= 40


# --- Enformer: exact windows only ---------------------------------------------------

def test_enformer_refuses_a_window_of_the_wrong_length():
    from data_loader.enformer_encoder import sequence_to_indices

    assert sequence_to_indices("ACGTN", length=5).tolist() == [0, 1, 2, 3, 4]
    for bad in ("ACGT", "ACGTNA"):
        with pytest.raises(ValueError):
            sequence_to_indices(bad, length=5)


# --- CPU smoke on the real models (slow; needs the cached hub snapshots) --------

def _snapshot(spec) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE  # honours HF_HOME / HF_HUB_CACHE

    name = "models--" + spec.model_name.replace("/", "--")
    return (Path(HF_HUB_CACHE) / name / "snapshots" / spec.revision).exists()


@pytest.mark.slow
def test_real_hyena_extraction_is_dna_only_and_not_constant(tmp_path, monkeypatch):
    from importlib import import_module

    from data_loader.model_registry import ENCODER_SPECS
    from data_loader.multi_pool import embed_all_multi_pool
    from data_loader.pooling_aggregator import aggregate
    from data_loader.sequence_fetcher import fetch_cds
    from linear_trainer.sources import check_not_constant

    spec = ENCODER_SPECS["hyena_dna"]
    if not _snapshot(spec) or not (DATA / "sequences").exists():
        pytest.skip("HyenaDNA snapshot or CDS cache not on this machine")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loader = import_module(spec.loader_module).load_model
    seen = []

    def load(device):
        model, tok, dev = loader("cpu")
        def record(input_ids, attention_mask=None):
            seen.append(input_ids[0].tolist())
            return model(input_ids=input_ids, attention_mask=attention_mask)
        return record, tok, dev

    genes = sorted(p.stem for p in (DATA / "sequences").glob("ENSG*.fa"))[:20]
    red = embed_all_multi_pool({g: fetch_cds(g, DATA / "sequences") for g in genes},
                               load, tmp_path, spec)
    tok = import_module(spec.loader_module).AutoTokenizer.from_pretrained(
        spec.model_name, revision=spec.revision, trust_remote_code=True)
    specials = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    assert seen and not any(set(ids) & specials for ids in seen)
    assert all(set(r) == {"mean", "max"} for r in red.values())
    check_not_constant(np.stack([aggregate(r, "meanmean") for r in red.values()]), "hyena smoke")


@pytest.mark.slow
def test_real_loaders_fill_every_weight_at_the_pinned_revision(monkeypatch):
    from importlib import import_module

    from data_loader.model_registry import ENCODER_SPECS

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = 0
    for spec in ENCODER_SPECS.values():
        if not _snapshot(spec):
            continue
        import_module(spec.loader_module).load_model("cpu")  # LoadError if anything is missing
        loaded += 1
    if not loaded:
        pytest.skip("no encoder snapshots on this machine")


def test_constant_check_ignores_zero_columns_and_one_large_constant_column():
    from linear_trainer.sources import ConstantFeatures, check_not_constant

    rng = np.random.default_rng(0)
    signal = rng.standard_normal((200, 20))
    sparse = np.hstack([signal, np.zeros((200, 300))])            # mostly all-zero columns
    check_not_constant(sparse, "sparse")
    check_not_constant(np.hstack([signal, np.full((200, 1), 1e6)]), "one big constant column")
    with pytest.raises(ConstantFeatures):
        check_not_constant(np.zeros((200, 5)), "zeros")
    with pytest.raises(ConstantFeatures):
        check_not_constant(np.ones((200, 5)) * 3.0 + 1e-9 * rng.standard_normal((200, 5)), "flat")


def test_cache_writes_are_atomic(tmp_path, monkeypatch):
    from data_loader import cache_meta

    def dies_midway(fh, **arrays):
        fh.write(b"PK\x03\x04 truncated")
        raise KeyboardInterrupt  # a spot interruption or OOM kill

    monkeypatch.setattr(cache_meta.np, "savez", dies_midway)
    with pytest.raises(KeyboardInterrupt):
        cache_meta.write_npz(tmp_path / "ENSG1.npz", {"mean": np.zeros(2)}, {"a": 1})
    assert not (tmp_path / "ENSG1.npz").exists()  # the resume sees no file, not a torn one


def test_reductions_from_another_device_are_refused_and_never_mixed(tmp_path):
    from data_loader.cache_meta import StaleCache, write_npz
    from data_loader.multi_pool import embed_all_multi_pool, extraction_meta, load_reductions

    def load(device):
        return _CausalModel(), _Tok(), device

    seqs = {"ENSG1": "ACGTAC", "ENSG2": "ACGGTA"}
    embed_all_multi_pool({"ENSG1": seqs["ENSG1"]}, load, tmp_path, _spec(), device="cpu")
    with pytest.raises(StaleCache):  # a CPU pilot's file is not reused by the GPU run
        embed_all_multi_pool({"ENSG1": seqs["ENSG1"]}, load, tmp_path, _spec(), device="cuda")
    write_npz(tmp_path / "ENSG2.npz", {"mean": np.zeros((1, 2)), "max": np.zeros((1, 2))},
              extraction_meta(_spec(), seqs["ENSG2"], "cuda"))
    with pytest.raises(StaleCache):  # a builder refuses a cache that mixes devices
        load_reductions(tmp_path, _spec(), seqs)
    assert set(load_reductions(tmp_path, _spec(), {"ENSG1": seqs["ENSG1"]})) == {"ENSG1"}


def test_enformer_cache_is_reused_only_when_built_the_same_way(tmp_path, monkeypatch):
    from data_loader import enformer_encoder as ee
    from data_loader.cache_meta import StaleCache

    loads = []
    monkeypatch.setattr(ee, "load_model", lambda device: loads.append(device) or (None, device))
    monkeypatch.setattr(ee, "extract_features", lambda seq, model, device, center_bins=16: {
        "trunk_global": np.ones(3, dtype=np.float32), "trunk_center": np.zeros(3, dtype=np.float32),
        "tracks_center": np.zeros(2, dtype=np.float32)})
    wins = {"ENSG1": "ACGT" * 8}
    ee.embed_all_enformer(wins, tmp_path, device="cpu")
    ee.embed_all_enformer(wins, tmp_path, device="cpu")
    assert loads == ["cpu"]  # the second call reused the cache
    with pytest.raises(StaleCache):
        ee.embed_all_enformer(wins, tmp_path, device="cpu", center_bins=8)
    with pytest.raises(StaleCache):
        ee.embed_all_enformer({"ENSG1": "ACGA" * 8}, tmp_path, device="cpu")


def test_esm2_dataset_refuses_embeddings_of_another_protein_or_run(tmp_path):
    import build_esm2_datasets as be
    from data_loader.cache_meta import StaleCache, sha256_text, write_npz
    from protein import translate_cds
    from synth import call_main

    seqs = tmp_path / "seqs"
    seqs.mkdir()
    cds = {"ENSGX1": "ATGAAATAAAAATGA", "ENSGX2": "ATGCCCGGGTGA"}  # not in the CDS manifest
    for g, s in cds.items():
        (seqs / f"{g}.fa").write_text(f">ENST_{g}.1\n{s}\n")
    pd.DataFrame({"ensembl_id": list(cds), "symbol": ["A", "B"], "family": ["tf", "ion"],
                  "summary": ["", ""], "y": [np.zeros(2, np.float32)] * 2}).to_parquet(
        tmp_path / "template.parquet")

    def write(g, protein, device="cuda"):
        write_npz(tmp_path / "emb" / f"{g}.npz", {"emb": np.ones(640, np.float32)},
                  {"model": "esm2_t30_150M_UR50D", "fp16": True, "device": device,
                   "max_residues": 1022, "translation": "through",
                   "protein_sha256": sha256_text(protein)})

    (tmp_path / "emb").mkdir()
    for g, s in cds.items():
        write(g, translate_cds(s, mode="through"))
    args = ["--size", "150m", "--template", str(tmp_path / "template.parquet"),
            "--emb-cache", str(tmp_path / "emb"), "--seq-cache", str(seqs),
            "--out", str(tmp_path / "out.parquet")]
    call_main(be, args)
    assert len(pd.read_parquet(tmp_path / "out.parquet")) == 2
    write("ENSGX1", "MK")  # the first-stop protein, not the full-length one
    with pytest.raises(StaleCache):
        call_main(be, args)
    write("ENSGX1", translate_cds(cds["ENSGX1"], mode="through"), device="cpu")
    with pytest.raises(StaleCache):  # one gene from another device
        call_main(be, args)
    (tmp_path / "emb" / "ENSGX1.npz").unlink()
    with pytest.raises(RuntimeError):
        call_main(be, args)
