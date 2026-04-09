"""Fetch CDS sequences from Ensembl REST and cache them as FASTA files.

Ensembl's `/sequence/id/{gene_id}?type=cds` is unreliable when given a *gene* ID,
so we do it in two steps:

    1. /lookup/id/{gene_id}        -> canonical_transcript (an ENST id)
    2. /sequence/id/{transcript_id}?type=cds  -> the CDS

Both steps are cached: lookups in `data/sequences/_lookup/{gene_id}.json`,
sequences in `data/sequences/{gene_id}.fa` (keyed by gene id, not transcript id,
so the rest of the pipeline keeps working).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

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


def fetch_cds(gene_id: str, cache_dir: str | Path) -> str | None:
    """Fetch the canonical CDS for an Ensembl gene ID. Cached on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{gene_id}.fa"

    if cache_file.exists():
        return _parse_fasta(cache_file.read_text())

    transcript_id = _canonical_transcript(gene_id, cache_dir / "_lookup")
    if not transcript_id:
        return None

    text = _request_fasta(SEQ_URL.format(transcript_id=transcript_id))
    if not text:
        return None

    cache_file.write_text(text)
    return _parse_fasta(text)


def fetch_all(gene_ids: list[str], cache_dir: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for gid in tqdm(gene_ids, desc="ensembl CDS"):
        seq = fetch_cds(gid, cache_dir)
        if seq:
            out[gid] = seq
        time.sleep(RATE_LIMIT_SLEEP)
    return out
