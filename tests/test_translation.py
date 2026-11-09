"""Guards for CDS translation (G5).

Every caller chooses a mode: the protein comparators translate through
(internal stops become X, the terminal stop is dropped), and only the frozen
MMseqs2 clustering keeps the first-stop truncation. CDSs that break the
"length = CDS/3 - 1, no internal stop" rule are listed, never silently cut.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXCEPTIONS = ROOT / "data" / "translation_exceptions.tsv"
SEQUENCES = ROOT / "data" / "sequences"

# ATG TAA AAA AAA TGA: an internal stop, then two lysines, then the real stop
INTERNAL_STOP = "ATGTAAAAAAAATGA"


def test_translate_requires_an_explicit_mode():
    from protein import translate_cds

    with pytest.raises(TypeError):
        translate_cds("ATGAAATGA")


def test_through_mode_keeps_full_length():
    from protein import translate_cds

    assert translate_cds(INTERNAL_STOP, mode="through") == "MXKK"
    assert translate_cds(INTERNAL_STOP, mode="first_stop") == "M"
    assert translate_cds("ATGAAATGA", mode="through") == "MK"
    with pytest.raises(ValueError):
        translate_cds("ATG", mode="to_stop")


def test_check_translation_names_each_failure():
    from protein import check_translation

    assert check_translation("ATGAAATGA") == []
    assert check_translation(INTERNAL_STOP) == ["internal_stop"]
    assert check_translation("ATGAAATGAC") == ["length_not_multiple_of_3"]
    assert check_translation("ATGAAAAAA") == ["no_terminal_stop"]
    assert check_translation("ATGNAATGA") == ["non_acgt"]


def test_aa_kmer_counts_residues_past_an_internal_stop():
    from composition_baseline.aa_kmer import featurize_aa_kmer
    from protein import AMINO_ACIDS

    v = featurize_aa_kmer(INTERNAL_STOP, k=1)
    assert v[AMINO_ACIDS.index("K")] == pytest.approx(2 / 3)
    assert v[AMINO_ACIDS.index("M")] == pytest.approx(1 / 3)


def test_clustering_fasta_keeps_the_frozen_first_stop_proteins(tmp_path):
    from cluster.mmseqs_cluster import write_protein_fasta

    seqs = tmp_path / "sequences"
    seqs.mkdir()
    (seqs / "ENSG1.fa").write_text(">ENST1.1\n" + INTERNAL_STOP + "\n")
    n, missing = write_protein_fasta(["ENSG1"], tmp_path / "p.fasta", sequences_dir=seqs)
    assert (n, missing) == (1, [])
    assert (tmp_path / "p.fasta").read_text() == ">ENSG1\nM\n"


def test_every_translate_call_names_its_mode():
    """Static scan: first_stop only in the frozen clustering path (and the
    exceptions list, which reports the truncated length)."""
    calls = []
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        text = path.read_text()
        for m in re.finditer(r"(?<!def )\btranslate_cds\(", text):
            depth, i = 1, m.end()
            while depth:  # the argument list, nested calls included
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            calls.append((path.relative_to(ROOT).as_posix(), text[m.end():i - 1]))
    assert calls, "scan found no calls"
    for where, args in calls:
        mode = re.search(r"mode=\"(\w+)\"", args)
        assert mode, f"{where}: translate_cds({args}) has no explicit mode"
        if mode.group(1) == "first_stop":
            assert where in ("src/cluster/mmseqs_cluster.py",
                             "scripts/list_translation_exceptions.py"), where


@pytest.mark.slow
def test_translation_failures_equal_the_tracked_list():
    from data_loader.sequence_fetcher import fetch_cds
    from protein import check_translation

    if not SEQUENCES.exists():
        pytest.skip("CDS cache not on this machine")
    listed = pd.read_csv(EXCEPTIONS, sep="\t")
    found = {}
    for path in sorted(SEQUENCES.glob("ENSG*.fa")):
        reasons = check_translation(fetch_cds(path.stem, SEQUENCES))
        if reasons:
            found[path.stem] = ";".join(reasons)
    assert found == dict(zip(listed["ensembl_id"], listed["reason"]))
    # 17 truncated by the first-stop rule (12 frameshifted, 5 in-frame stops),
    # plus one CDS with a non-ACGT base that translates to X at full length
    truncated = listed["len_through"] > listed["len_first_stop"]
    assert truncated.sum() == 17
    assert set(listed.loc[~truncated, "reason"]) == {"non_acgt"}


# --- G19: the CDS inputs are pinned too -------------------------------------------

def test_a_missing_cds_raises_instead_of_fetching_todays_release(tmp_path, monkeypatch):
    from data_loader import sequence_fetcher as sf

    def no_network(*a, **k):
        raise AssertionError("fetch_cds went to the network")

    monkeypatch.setattr(sf, "_canonical_transcript", no_network)
    monkeypatch.setattr(sf, "_request_fasta", no_network)
    with pytest.raises(sf.MissingCDS):
        sf.fetch_cds("ENSG00000000938", tmp_path)


def test_a_cds_that_differs_from_the_manifest_is_refused(tmp_path):
    from data_loader import sequence_fetcher as sf

    m = pd.read_csv(ROOT / "data" / "cds_manifest.tsv", sep="\t").set_index("ensembl_id")
    gene = m.index[0]
    real = (SEQUENCES / f"{gene}.fa").read_text() if SEQUENCES.exists() else None
    (tmp_path / f"{gene}.fa").write_text(f">{m.at[gene, 'transcript_id']}\nATGAAATGA\n")
    with pytest.raises(sf.StaleCDS):  # another sequence under the right transcript
        sf.fetch_cds(gene, tmp_path)
    if real is not None:
        other = real.replace(m.at[gene, "transcript_id"], "ENST00000000000.1", 1)
        (tmp_path / f"{gene}.fa").write_text(other)
        with pytest.raises(sf.StaleCDS):  # the right sequence under another transcript
            sf.fetch_cds(gene, tmp_path)
        (tmp_path / f"{gene}.fa").write_text(real)
        assert sf.fetch_cds(gene, tmp_path) == sf._parse_fasta(real)


def test_a_network_fetch_of_another_release_is_refused_before_it_is_cached(tmp_path, monkeypatch):
    from data_loader import sequence_fetcher as sf

    m = pd.read_csv(ROOT / "data" / "cds_manifest.tsv", sep="\t").set_index("ensembl_id")
    gene = m.index[0]
    stable = m.at[gene, "transcript_id"].split(".")[0]
    monkeypatch.setattr(sf, "_canonical_transcript", lambda g, d: stable)
    monkeypatch.setattr(sf, "_request_fasta", lambda url: f">{stable}.99\nATGAAATGA\n")
    with pytest.raises(sf.StaleCDS):  # today's REST serves another version
        sf.fetch_cds(gene, tmp_path, allow_network=True)
    assert not (tmp_path / f"{gene}.fa").exists()


def test_a_missing_cds_manifest_is_an_error(tmp_path, monkeypatch):
    from data_loader import sequence_fetcher as sf

    monkeypatch.setattr(sf, "CDS_MANIFEST", tmp_path / "absent.tsv")
    sf._manifest.cache_clear()
    try:
        (tmp_path / "ENSG1.fa").write_text(">ENST1.1\nATGAAATGA\n")
        with pytest.raises(FileNotFoundError):
            sf.fetch_cds("ENSG1", tmp_path)
    finally:
        sf._manifest.cache_clear()
