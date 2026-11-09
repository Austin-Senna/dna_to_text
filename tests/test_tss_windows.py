"""Guards for the strand-aware canonical-TSS windows and the TSS split (G2, G4, G19, G21).

Every window is exactly L bases in gene orientation with the transcript's 5' end
at index L//2; chromosome edges are N-padded, never shifted. Windows are read
only through ``read_window``, which checks each file against the manifest.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from synth import call_main, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def _chrom(n: int = 100, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _fwd(chrom: str, span) -> str:
    return chrom[span.start - 1:span.end]


def _rc(s: str) -> str:
    return "".join(COMP[b] for b in reversed(s))


def _window(chrom: str, tss: int, strand: int, length: int = 11) -> str:
    from data_loader.enformer_windows import orient, window_span

    span = window_span("1", tss, strand, len(chrom), length=length)
    return orient(_fwd(chrom, span), span)


# --- G4: strand and centring -------------------------------------------------

def test_minus_strand_window_is_reverse_complemented_with_tss_at_centre():
    chrom = _chrom()
    tss, k = 60, 5  # 1-based 5' end of a minus-strand transcript
    win = _window(chrom, tss, -1)
    # transcript reads 5'->3' toward lower coordinates, on the minus strand
    transcript_prefix = _rc(chrom[tss - k:tss])
    assert len(win) == 11
    assert win[11 // 2:11 // 2 + k] == transcript_prefix
    assert win == _rc(chrom[tss - 6:tss + 5])


def test_plus_strand_window_has_tss_at_centre():
    chrom = _chrom()
    tss, k = 40, 5
    win = _window(chrom, tss, 1)
    assert win[11 // 2:11 // 2 + k] == chrom[tss - 1:tss - 1 + k]
    assert win == chrom[tss - 6:tss + 5]


@pytest.mark.parametrize("tss,strand,pads", [
    (2, 1, ("up", 4)),     # left edge, plus: upstream runs off the chromosome
    (3, -1, ("down", 3)),  # left edge, minus: downstream runs off
    (98, 1, ("down", 3)),  # right edge, plus
    (99, -1, ("up", 4)),   # right edge, minus
])
def test_edge_windows_are_n_padded_not_shifted(tss, strand, pads):
    from data_loader.enformer_windows import window_span

    chrom = _chrom()
    span = window_span("1", tss, strand, len(chrom), length=11)
    win = _window(chrom, tss, strand)
    assert len(win) == 11
    first = chrom[tss - 1] if strand > 0 else COMP[chrom[tss - 1]]
    assert win[11 // 2] == first
    side, n = pads
    assert (span.pad_up, span.pad_down) == ((n, 0) if side == "up" else (0, n))
    assert win.startswith("N" * n) if side == "up" else win.endswith("N" * n)
    assert 1 <= span.start <= span.end <= len(chrom)


def test_real_length_puts_tss_at_98304():
    from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, TSS_INDEX, window_span

    assert ENFORMER_WINDOW_LENGTH == 196_608 and TSS_INDEX == 98_304
    for strand in (1, -1):
        span = window_span("7", 5_000_000, strand, 159_345_973)
        assert span.end - span.start + 1 == ENFORMER_WINDOW_LENGTH
        idx = (5_000_000 - span.start) if strand > 0 else (span.end - 5_000_000)
        assert idx == TSS_INDEX


def test_reverse_complement_handles_iupac_and_refuses_unknown():
    from data_loader.enformer_windows import reverse_complement

    assert reverse_complement("ACGTNRYKM") == "KMRYNACGT"
    with pytest.raises(ValueError):
        reverse_complement("ACGU")


# --- local FASTA access -------------------------------------------------------

def test_fasta_index_reads_across_line_breaks(tmp_path):
    from data_loader.enformer_windows import FastaIndex

    a, b = _chrom(95, seed=1), _chrom(23, seed=2)
    wrap = lambda s: "\n".join(s[i:i + 10] for i in range(0, len(s), 10))
    fa = tmp_path / "g.fa"
    fa.write_text(f">1 dna:chromosome\n{wrap(a)}\n>MT extra words\n{wrap(b)}\n")
    idx = FastaIndex(fa)
    assert idx.lengths() == {"1": 95, "MT": 23}
    assert idx.fetch("1", 1, 95) == a
    assert idx.fetch("1", 9, 31) == a[8:31]
    assert idx.fetch("MT", 20, 23) == b[19:23]
    with pytest.raises(ValueError):
        idx.fetch("1", 90, 96)

    bad = tmp_path / "bad.fa"
    bad.write_text(">1\nACGTACGTAC\nACG\nACGTACGTAC\n")
    with pytest.raises(ValueError):
        FastaIndex(bad)


def test_cdna_prefixes_come_from_the_named_transcript_version(tmp_path):
    from data_loader.enformer_windows import read_cdna_prefixes

    fa = tmp_path / "cdna.fa"
    fa.write_text(">ENST1.2 cdna x\nACGTA\nCCGGT\n>ENST1.3 cdna x\nTTTTT\n>ENST9.1 cdna\nGG\n")
    assert read_cdna_prefixes(fa, {"ENST1.2", "ENST9.1"}, 7) == {"ENST1.2": "ACGTACC", "ENST9.1": "GG"}


# --- G19: the manifest is the only way in -----------------------------------

def _write_manifest(tmp: Path, seqs: dict[str, str]) -> Path:
    from data_loader.enformer_windows import write_window

    rows = []
    for gid, seq in seqs.items():
        rows.append(write_window(tmp / "windows", gid, seq, {
            "symbol": gid, "transcript_id": "ENST0.1", "chrom": "1", "strand": 1,
            "tss": 6, "start": 1, "end": len(seq), "pad_up": 0, "pad_down": 0,
            "chrom_len": 100}))
    manifest = tmp / "manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest, sep="\t", index=False)
    return manifest


def test_read_window_verifies_the_manifest(tmp_path):
    from data_loader.enformer_windows import StaleWindow, read_window

    chrom = _chrom()
    seqs = {"ENSG1": chrom[:11], "ENSG2": chrom[20:31]}
    manifest = _write_manifest(tmp_path, seqs)
    kw = dict(manifest=manifest, window_dir=tmp_path / "windows", length=11)
    assert read_window("ENSG1", **kw) == seqs["ENSG1"]

    # an old-geometry file under the same name is refused, not silently used
    path = tmp_path / "windows" / "ENSG2.fa"
    path.write_text(">ENSG2\n" + chrom[21:32] + "\n")
    with pytest.raises(StaleWindow):
        read_window("ENSG2", **kw)
    with pytest.raises(StaleWindow):
        read_window("ENSG3", **kw)


# --- G2: the leak statistic is computed, not quoted ----------------------------

def _old_spans() -> dict[str, tuple[str, int, int]]:
    """The May windows' spans, read from git: scripts/tss_overlap.py now rewrites
    the working copy of this table from the new manifest."""
    import io
    import subprocess

    res = subprocess.run(
        ["git", "-C", str(ROOT), "show", "095dcf5:analysis/tss_overlap/tables/per_gene_overlap.csv"],
        capture_output=True, text=True)
    if res.returncode:
        pytest.skip("commit 095dcf5 not in this clone (shallow?)")
    text = res.stdout
    df = pd.read_csv(io.StringIO(text), usecols=["ensembl_id", "chrom", "window_start", "window_end"],
                     dtype={"chrom": str})
    return {r.ensembl_id: (r.chrom, int(r.window_start), int(r.window_end))
            for r in df.itertuples(index=False)}


def test_window_leak_stats_reproduce_the_audit_on_the_old_windows():
    from splits.window_leak import window_leak_stats

    spans = _old_spans()
    split = json.loads((DATA / "splits.json").read_text())
    assert set(spans) == set(split["train"]) | set(split["val"]) | set(split["test"])
    s = window_leak_stats(split, spans)
    assert s["cross_split_pairs"] == 922
    assert (s["pairs_test_train"], s["pairs_train_val"], s["pairs_test_val"]) == (404, 407, 111)
    assert (s["n_test"], s["test_overlapping_trainval"]) == (487, 235)
    assert (s["test_overlapping_train"], s["test_overlapping_val"]) == (192, 93)
    assert s["trainval_genes_overlapping_test"] == 449
    assert round(100 * s["frac_test_overlapping_trainval"], 1) == 48.3


# --- G21 + G2: the disjoint split builder --------------------------------------

PLANTED = ((5, 16), (40, 71), (1, 30), (8, 57), (12, 88), (24, 46), (33, 64), (49, 79))


def _toy(n: int = 90):
    """Sparse windows (no overlaps) with planted ones, protein triplets, and one
    cluster whose representative is outside the gene set."""
    genes = [f"ENSG{i:011d}" for i in range(n)]
    fams = pd.DataFrame({"ensembl_id": genes,
                         "family": [("gpcr", "ion", "tf")[i % 3] for i in range(n)]})
    spans = {g: ("1", 1 + 10_000 * i, 1_000 + 10_000 * i) for i, g in enumerate(genes)}
    for a, b in PLANTED:  # planted overlaps across triplets
        c, s0, _ = spans[genes[a]]
        spans[genes[b]] = (c, s0 + 500, s0 + 1_499)
    protein_map = {g: genes[(i // 3) * 3] for i, g in enumerate(genes)}
    for i in (20, 52, 83):  # rep outside the universe: members must still share a split
        protein_map[genes[i]] = "ENSG_OUTSIDE"
    return fams, spans, protein_map


def test_disjoint_split_ignores_row_order_and_keeps_every_group_together():
    from splits.tss_disjoint import build_tss_disjoint
    from splits.window_leak import window_leak_stats

    fams, spans, pmap = _toy()
    a, stats = build_tss_disjoint(fams, spans, pmap, seed=42)
    b, _ = build_tss_disjoint(fams.sample(frac=1, random_state=7), spans, pmap, seed=42)
    assert all(len(a[k]) >= 10 for k in ("train", "val", "test")), stats  # a real split
    assert {k: a[k] for k in ("train", "val", "test")} == {k: b[k] for k in ("train", "val", "test")}
    assert window_leak_stats(a, spans)["cross_split_pairs"] == 0
    split_of = {g: s for s in ("train", "val", "test") for g in a[s]}
    genes = fams["ensembl_id"].tolist()
    for i, j in PLANTED:
        assert split_of[genes[i]] == split_of[genes[j]]
    by_rep: dict = {}
    for member, rep in pmap.items():
        by_rep.setdefault(rep, set()).add(split_of[member])
    assert all(len(v) == 1 for v in by_rep.values()), "a protein cluster straddles splits"


def test_disjoint_split_reads_the_tracked_clusters_and_stamps_inputs(tmp_path, monkeypatch):
    import make_tss_disjoint_split as mk

    assert mk.CLUSTER_TSV == DATA / "clusters" / "homology_id40.tsv"

    fams, spans, pmap = _toy()
    fam_parquet = tmp_path / "fams.parquet"
    fams.to_parquet(fam_parquet)
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{"ensembl_id": g, "chrom": c, "start": s, "end": e}
                  for g, (c, s, e) in spans.items()]).to_csv(manifest, sep="\t", index=False)
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("".join(f"{rep}\t{m}\n" for m, rep in pmap.items()))
    genes = fams["ensembl_id"].tolist()
    universe = tmp_path / "splits.json"
    universe.write_text(json.dumps({"train": genes[:60], "val": genes[60:75], "test": genes[75:]}))
    out, leak = tmp_path / "out.json", tmp_path / "leak.json"
    call_main(mk, ["--manifest", str(manifest), "--clusters", str(clusters),
                   "--families", str(fam_parquet), "--splits", str(universe),
                   "--out", str(out), "--leak-out", str(leak), "--seed", "3"])
    # inputs inside the repo are stamped repo-relative: the split file is tracked (G22)
    assert mk._stamp(DATA / "splits.json")["path"] == "data/splits.json"
    payload = json.loads(out.read_text())
    assert payload["seed"] == 3
    assert payload["inputs"] == {
        "window_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "cluster_tsv": {"path": str(clusters), "sha256": sha256_file(clusters)},
        "gene_universe": {"path": str(universe), "sha256": sha256_file(universe)},
        "families": {"path": str(fam_parquet), "labels_sha256": mk._labels_stamp(fam_parquet, fams)["labels_sha256"]},
    }
    stats = json.loads(leak.read_text())
    assert stats[out.name]["cross_split_pairs"] == 0
    assert universe.name in stats

    # a seed split can never land on the primary split's path
    with pytest.raises(ValueError):
        call_main(mk, ["--manifest", str(manifest), "--clusters", str(clusters),
                       "--families", str(fam_parquet), "--splits", str(universe), "--seed", "7"])


# --- G4 on the real windows (slow; needs the local window cache) ---------------

MANIFEST = DATA / "tss_windows.tsv"


@pytest.mark.slow
def test_cached_windows_start_with_the_transcript_cdna():
    """Every window, from its TSS on, reads the transcript's own cDNA prefix.

    The prefixes in the manifest came from the release's cDNA FASTA, not from
    the genome, so this re-checks centring and strand on every cached window.
    """
    from data_loader.enformer_windows import (
        ENFORMER_WINDOW_LENGTH,
        TSS_INDEX,
        WINDOW_DIR,
        load_manifest,
        read_window,
    )

    if not WINDOW_DIR.exists():
        pytest.skip("window cache not built on this machine")
    m = load_manifest(MANIFEST)
    assert len(m) == 3244 and set(m["strand"]) == {1, -1}
    assert int(((m["pad_up"] + m["pad_down"]) > 0).sum()) == 8  # the chromosome-edge genes
    assert (m["end"] - m["start"] + 1 + m["pad_up"] + m["pad_down"] == ENFORMER_WINDOW_LENGTH).all()
    for gene, r in m.iterrows():
        win = read_window(gene)
        assert win[TSS_INDEX:TSS_INDEX + len(r["cdna_prefix"])] == r["cdna_prefix"], r["symbol"]


# --- G19: E5 offsets must come from the current windows and tokenizer ----------

def test_e5_offsets_are_checked_against_the_windows_and_tokenizer():
    from check_tss_center_chunk import verify_offsets
    from data_loader.cache_meta import StaleCache
    from data_loader.enformer_windows import load_manifest
    from data_loader.model_registry import ENCODER_SPECS

    spec = ENCODER_SPECS["nt_v2"]
    m = load_manifest(MANIFEST)
    genes = list(m.index[:3])
    good = pd.DataFrame({"ensembl_id": genes, "tss_chunk_idx": [1, 2, 3],
                         "window_sha256": [m.at[g, "sha256"] for g in genes],
                         "tokenizer_revision": spec.revision}).set_index("ensembl_id")
    verify_offsets(good, spec, genes)
    with pytest.raises(StaleCache):  # the tracked May CSVs carry no provenance
        verify_offsets(good.drop(columns=["window_sha256"]), spec, genes)
    stale = good.copy()
    stale.loc[genes[1], "window_sha256"] = "0" * 64
    with pytest.raises(StaleCache):
        verify_offsets(stale, spec, genes)
    with pytest.raises(StaleCache):
        verify_offsets(good.assign(tokenizer_revision="abc"), spec, genes)
    with pytest.raises(StaleCache):
        verify_offsets(good.drop(index=genes[2]), spec, genes)
