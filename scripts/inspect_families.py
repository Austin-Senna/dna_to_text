"""Preview each family's HGNC matches (and intersection with GenePT) before
running the full pipeline.

Usage:
    python scripts/inspect_families.py
    python scripts/inspect_families.py kinase
    python scripts/inspect_families.py kinase tf
    python scripts/inspect_families.py --no-genept     # skip the GenePT join
    python scripts/inspect_families.py --show 30       # show 30 sample symbols
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_loader.dataset_loader import (
    FAMILIES,
    filter_family,
    load_genept_embeddings,
    load_hgnc_complete,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
GENEPT_PICKLE = REPO_ROOT / "GenePT_emebdding_v2" / "GenePT_gene_embedding_ada_text.pickle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", help="Subset of family short names")
    ap.add_argument("--no-genept", action="store_true")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    print("loading HGNC complete set...")
    hgnc = load_hgnc_complete(DATA / "hgnc")
    print(f"  {len(hgnc):,} protein-coding genes")

    embed_keys: set[str] = set()
    if not args.no_genept:
        print("loading GenePT keys...")
        embed_keys = set(load_genept_embeddings(GENEPT_PICKLE).keys())
        print(f"  {len(embed_keys):,} GenePT genes")

    selected = FAMILIES
    if args.families:
        wanted = set(args.families)
        selected = [f for f in FAMILIES if f[0] in wanted]

    for short, display, includes, excludes in selected:
        print(f"\n=== {display} ({short}) ===")
        print(f"  include: {includes}")
        if excludes:
            print(f"  exclude: {excludes}")
        fam = filter_family(hgnc, includes, excludes)
        fam = fam.dropna(subset=["ensembl_id"])
        print(f"  HGNC matches              : {len(fam):5d}")
        if embed_keys:
            inter = fam[fam["symbol"].isin(embed_keys)]
            print(f"  ∩ GenePT (with ensembl_id): {len(inter):5d}")
            sample = inter["symbol"].head(args.show).tolist()
            print(f"  sample symbols ({len(sample)}): {sample}")
            # show a few raw gene_group strings to sanity-check the regex
            print("  sample gene_group strings:")
            for s in inter["gene_group"].head(5):
                print(f"    {s[:140]}")
        else:
            sample = fam["symbol"].head(args.show).tolist()
            print(f"  sample symbols ({len(sample)}): {sample}")


if __name__ == "__main__":
    main()
