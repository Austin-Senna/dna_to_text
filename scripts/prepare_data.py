"""CPU-only step: GenePT + HGNC + Ensembl CDS fetch.

Produces an intermediate parquet with everything except the DNABERT-2 vectors.
Run this on a laptop; the encoder step picks up where this leaves off.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_loader.dataset_loader import (
    FAMILIES,
    analyze_genept,
    build_gene_table,
)
from data_loader.sequence_fetcher import fetch_all

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
GENEPT_DIR = REPO_ROOT / "GenePT_emebdding_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genept-pickle", default=str(GENEPT_DIR / "GenePT_gene_embedding_ada_text.pickle"))
    ap.add_argument("--genept-summaries", default=str(GENEPT_DIR / "NCBI_summary_of_genes.json"))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--families", default=None, help="Comma-separated short names: kinase,tf,ion,gpcr,immune")
    ap.add_argument("--limit", type=int, default=None, help="Per-family gene cap")
    ap.add_argument("--out", default=str(DATA / "gene_table.parquet"))
    args = ap.parse_args()

    print("=== analyze GenePT ===")
    analyze_genept(args.genept_pickle)
    if args.analyze_only:
        return

    families = FAMILIES
    if args.families:
        wanted = set(args.families.split(","))
        families = [f for f in FAMILIES if f[1] in wanted]

    print("\n=== build gene table ===")
    df = build_gene_table(
        args.genept_pickle,
        args.genept_summaries,
        DATA / "hgnc",
        families=families,
        per_family_limit=args.limit,
    )
    print(f"  total: {len(df)} genes")

    print("\n=== fetch CDS from Ensembl ===")
    cds = fetch_all(df["ensembl_id"].tolist(), DATA / "sequences")
    df = df[df["ensembl_id"].isin(cds.keys())].reset_index(drop=True)
    print(f"  with CDS: {len(df)} genes")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\n  wrote {out_path}")
    print("  per-family counts:")
    print(df["family"].value_counts().to_string())


if __name__ == "__main__":
    main()
