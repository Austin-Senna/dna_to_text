"""Fetch CDS sequences from Ensembl REST and cache them as FASTA files.

Ensembl's `/sequence/id/{gene_id}?type=cds` is unreliable when given a *gene* ID,
so we do it in two steps:

    1. /lookup/id/{gene_id}        -> canonical_transcript (an ENST id)
    2. /sequence/id/{transcript_id}?type=cds  -> the CDS

Both steps are cached: lookups in `data/sequences/_lookup/{gene_id}.json`,
sequences in `data/sequences/{gene_id}.fa` (keyed by gene id, not transcript id,
so the rest of the pipeline keeps working).

The cache is the pinned input (G19). The CDSs were fetched in May 2026, when
REST served Ensembl 115; REST now serves a later release in which some
canonical transcripts changed. So ``fetch_cds`` reads the cache only, and
checks each gene against the tracked ``data/cds_manifest.tsv`` (transcript
version and sha256). Only ``fetch_all``, the stage-1 dataset build, may go to
the network.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

CDS_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "cds_manifest.tsv"

LOOKUP_URL = "https://rest.ensembl.org/lookup/id/{gene_id}?expand=0"
SEQ_URL = "https://rest.ensembl.org/sequence/id/{transcript_id}?type=cds"
RATE_LIMIT_SLEEP = 1 / 14  # stay safely under 15 req/s


def _parse_fasta(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines).upper()


def _request_json(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 404):
            return None
        time.sleep(2 ** attempt)
    return None


def _request_fasta(url: str) -> str | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"Accept": "text/x-fasta"}, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code in (400, 404):
            return None
        time.sleep(2 ** attempt)
    return None


def _canonical_transcript(gene_id: str, lookup_dir: Path) -> str | None:
    lookup_dir.mkdir(parents=True, exist_ok=True)
    cache = lookup_dir / f"{gene_id}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        data = _request_json(LOOKUP_URL.format(gene_id=gene_id))
        if data is None:
            return None
        cache.write_text(json.dumps(data))
        time.sleep(RATE_LIMIT_SLEEP)

    # Ensembl returns canonical_transcript like "ENST00000338591.10"
    canon = data.get("canonical_transcript")
    if not canon:
        return None
    return canon.split(".")[0]  # strip version


class MissingCDS(RuntimeError):
    """A CDS is not in the cache and the network was not allowed."""


class StaleCDS(RuntimeError):
    """A cached CDS differs from the pinned manifest."""


@lru_cache(maxsize=1)
def _manifest() -> dict[str, tuple[str, str]]:
    if not CDS_MANIFEST.exists():  # never fall back to "no pinning"
        raise FileNotFoundError(f"{CDS_MANIFEST} is missing; it pins every CDS (G19)")
    m = pd.read_csv(CDS_MANIFEST, sep="\t")
    return {g: (t, h) for g, t, h in zip(m["ensembl_id"], m["transcript_id"], m["sha256"])}


def cds_sha256(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def _check_pinned(gene_id: str, text: str, where) -> None:
    """A gene in the manifest must carry its pinned transcript version and sequence."""
    pinned = _manifest().get(gene_id)
    if pinned is None:
        return
    transcript = text.split("\n", 1)[0][1:].split()[0]
    if (transcript, cds_sha256(_parse_fasta(text))) != pinned:
        raise StaleCDS(f"{where}: {transcript} does not match the pinned CDS {pinned[0]} "
                       "(data/cds_manifest.tsv)")


def fetch_cds(gene_id: str, cache_dir: str | Path, *, allow_network: bool = False) -> str | None:
    """The canonical CDS for an Ensembl gene ID, from the cache.

    A gene in ``data/cds_manifest.tsv`` must match its transcript version and
    sha256 (``StaleCDS``). A gene missing from the cache raises ``MissingCDS``
    unless ``allow_network``, which only the stage-1 build passes.
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{gene_id}.fa"

    if cache_file.exists():
        text = cache_file.read_text()
        _check_pinned(gene_id, text, cache_file)
        return _parse_fasta(text)
    if not allow_network:
        raise MissingCDS(f"no cached CDS for {gene_id} in {cache_dir}; copy data/sequences/ "
                         "from the machine that built the dataset (REST serves a later release)")
    cache_dir.mkdir(parents=True, exist_ok=True)

    transcript_id = _canonical_transcript(gene_id, cache_dir / "_lookup")
    if not transcript_id:
        return None

    text = _request_fasta(SEQ_URL.format(transcript_id=transcript_id))
    if not text:
        return None

    _check_pinned(gene_id, text, f"REST {transcript_id}")  # before it can enter the cache
    cache_file.write_text(text)
    return _parse_fasta(text)


def fetch_all(gene_ids: list[str], cache_dir: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for gid in tqdm(gene_ids, desc="ensembl CDS"):
        seq = fetch_cds(gid, cache_dir, allow_network=True)
        if seq:
            out[gid] = seq
        time.sleep(RATE_LIMIT_SLEEP)
    return out
