"""List the CDSs that break "protein length = CDS/3 - 1, no internal stop" (G5).

The protein comparators (AA k-mer baselines, ESM-2) translate these at full
length with X for internal stops; the frozen MMseqs2 clustering behind
data/splits.json used the first-stop proteins. tests/test_translation.py checks
this list against the CDS cache, so a new failure cannot slip through.

Run: uv run scripts/list_translation_exceptions.py
Writes: data/translation_exceptions.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from linear_trainer.sources import META_PARQUET
from protein import check_translation, translate_cds

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--out", type=Path, default=DATA / "translation_exceptions.tsv")
    args = ap.parse_args()

    symbols = pd.read_parquet(META_PARQUET, columns=["ensembl_id", "symbol"])
    symbol_of = dict(zip(symbols["ensembl_id"], symbols["symbol"]))
    rows = []
    for path in sorted(args.seq_cache.glob("ENSG*.fa")):
        cds = fetch_cds(path.stem, args.seq_cache)
        reasons = check_translation(cds)
        if not reasons:
            continue
        rows.append({
            "ensembl_id": path.stem,
            "symbol": symbol_of.get(path.stem, ""),
            "transcript_id": path.read_text().splitlines()[0][1:].split()[0],
            "reason": ";".join(reasons),
            "cds_len": len(cds),
            "len_first_stop": len(translate_cds(cds, mode="first_stop")),
            "len_through": len(translate_cds(cds, mode="through")),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {args.out.name}: {len(out)} genes")


if __name__ == "__main__":
    main()
