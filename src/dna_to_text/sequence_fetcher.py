"""Fetch CDS sequences from Ensembl REST and cache them as FASTA files."""
from __future__ import annotations

import time
from pathlib import Path

import requests
from tqdm import tqdm

ENSEMBL_URL = "https://rest.ensembl.org/sequence/id/{ensembl_id}?type=cds"
RATE_LIMIT_SLEEP = 1 / 14  # stay safely under 15 req/s


def _parse_fasta(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines).upper()


def fetch_cds(ensembl_id: str, cache_dir: str | Path) -> str | None:
    """Fetch the CDS for an Ensembl gene ID. Cached on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ensembl_id}.fa"

    if cache_file.exists():
        return _parse_fasta(cache_file.read_text())

    url = ENSEMBL_URL.format(ensembl_id=ensembl_id)
    headers = {"Accept": "text/x-fasta"}

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            cache_file.write_text(r.text)
            return _parse_fasta(r.text)
        if r.status_code in (400, 404):
            return None
        time.sleep(2 ** attempt)
    return None


def fetch_all(ensembl_ids: list[str], cache_dir: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for eid in tqdm(ensembl_ids, desc="ensembl CDS"):
        seq = fetch_cds(eid, cache_dir)
        if seq:
            out[eid] = seq
        time.sleep(RATE_LIMIT_SLEEP)
    return out
