"""Pin the CDS inputs: transcript version and sha256 per gene (G19).

The CDS cache (``data/sequences/``, gitignored) was fetched from Ensembl REST in
May 2026 (release 115). ``sequence_fetcher.fetch_cds`` checks every read against
this manifest, so a cache rebuilt from a later release, or a file edited by
hand, is refused instead of silently embedded.

Run: uv run scripts/build_cds_manifest.py
Writes: data/cds_manifest.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_loader.sequence_fetcher import _parse_fasta, cds_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=DATA / "splits.json",
                    help="gene universe (train + val + test)")
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--out", type=Path, default=DATA / "cds_manifest.tsv")
    args = ap.parse_args()

    universe = json.loads(args.splits.read_text())
    rows = []
    for g in sorted(x for s in ("train", "val", "test") for x in universe[s]):
        text = (args.seq_cache / f"{g}.fa").read_text()
        seq = _parse_fasta(text)
        rows.append({"ensembl_id": g, "transcript_id": text.split("\n", 1)[0][1:].split()[0],
                     "cds_len": len(seq), "sha256": cds_sha256(seq)})
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out.name}: {len(rows)} CDSs")


if __name__ == "__main__":
    main()
