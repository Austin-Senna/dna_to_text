"""Fetch and cache TSS-centered windows for Enformer features."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id/{gene_id}?expand=0"
ENSEMBL_REGION_URL = "https://rest.ensembl.org/sequence/region/human/{region}"
ENFORMER_WINDOW_LENGTH = 196_608
RATE_LIMIT_SLEEP = 1 / 14


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
            r = requests.get(url, headers={"Accept": "text/x-fasta"}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code in (400, 404):
            return None
        time.sleep(2 ** attempt)
    return None


def centered_window(
    seq_region_name: str,
    start: int,
    end: int,
    strand: int,
    length: int = ENFORMER_WINDOW_LENGTH,
) -> tuple[str, int, int]:
    """Return a 1-based inclusive TSS-centered Ensembl region."""
    if length <= 0:
        raise ValueError("length must be positive")
    tss = start if strand >= 0 else end
    left = length // 2
    right = length - left - 1
    region_start = tss - left
    region_end = tss + right
    if region_start < 1:
        region_end += 1 - region_start
        region_start = 1
    return seq_region_name, int(region_start), int(region_end)


def _lookup_gene(gene_id: str, lookup_dir: Path) -> dict | None:
    lookup_dir.mkdir(parents=True, exist_ok=True)
    cache = lookup_dir / f"{gene_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = _request_json(ENSEMBL_LOOKUP_URL.format(gene_id=gene_id))
    if data is None:
        return None
    cache.write_text(json.dumps(data))
    time.sleep(RATE_LIMIT_SLEEP)
    return data


def fetch_tss_window(
    gene_id: str,
    cache_dir: str | Path,
    length: int = ENFORMER_WINDOW_LENGTH,
) -> str | None:
    """Fetch a TSS-centered reference-genome window for an Ensembl gene ID."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{gene_id}.fa"
    if cache_file.exists():
        return _parse_fasta(cache_file.read_text())

    data = _lookup_gene(gene_id, cache_dir / "_lookup")
    if data is None:
        return None
    chrom, region_start, region_end = centered_window(
        seq_region_name=str(data["seq_region_name"]),
        start=int(data["start"]),
        end=int(data["end"]),
        strand=int(data.get("strand", 1)),
        length=length,
    )
    region = f"{chrom}:{region_start}..{region_end}:1"
    text = _request_fasta(ENSEMBL_REGION_URL.format(region=region))
    if not text:
        return None
    cache_file.write_text(text)
    return _parse_fasta(text)


def fetch_all_tss_windows(
    gene_ids: list[str],
    cache_dir: str | Path,
    length: int = ENFORMER_WINDOW_LENGTH,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for gid in tqdm(gene_ids, desc="ensembl TSS windows"):
        seq = fetch_tss_window(gid, cache_dir, length=length)
        if seq:
            out[gid] = seq
        time.sleep(RATE_LIMIT_SLEEP)
    return out
