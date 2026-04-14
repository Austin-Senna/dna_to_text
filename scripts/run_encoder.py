"""GPU step: load gene_table.parquet, embed CDS with DNABERT-2, write dataset.parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from data_loader.encoder_runner import embed_all

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-table", default=str(DATA / "gene_table.parquet"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--out", default=str(DATA / "dataset.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.gene_table)
    print(f"loaded {len(df)} genes from {args.gene_table}")

    # Read cached CDS from disk (prepare_data already fetched them)
    cds: dict[str, str] = {}
    for eid in df["ensembl_id"]:
        seq = fetch_cds(eid, DATA / "sequences")
        if seq:
            cds[eid] = seq
    df = df[df["ensembl_id"].isin(cds.keys())].reset_index(drop=True)
    print(f"  with CDS on disk: {len(df)}")

    print("\n=== embed with DNABERT-2 ===")
    device = None if args.device == "auto" else args.device
    x_vecs = embed_all(cds, DATA / "embeddings", device=device)
    df["x"] = df["ensembl_id"].map(lambda e: x_vecs[e])
    df = df.rename(columns={"y_embedding": "y"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\n  wrote {out_path} ({len(df)} rows)")
    print("  per-family counts:")
    print(df["family"].value_counts().to_string())


if __name__ == "__main__":
    main()
