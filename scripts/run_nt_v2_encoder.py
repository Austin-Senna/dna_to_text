"""Run NT-v2 100M multi-species over cached CDS and write dataset_nt_v2.parquet.

Reuses the gene list and y column from data/dataset.parquet so splits and
eval stay apples-to-apples with the DNABERT-2 run. Only the x column is
replaced with NT-v2 vectors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader.nt_v2_encoder import embed_all
from data_loader.sequence_fetcher import fetch_cds

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-in", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--cache-dir", default=str(DATA / "embeddings_nt_v2"))
    ap.add_argument("--out", default=str(DATA / "dataset_nt_v2.parquet"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset_in)
    print(f"loaded {len(df)} genes from {args.dataset_in}")

    cds: dict[str, str] = {}
    for eid in df["ensembl_id"]:
        seq = fetch_cds(eid, DATA / "sequences")
        if seq:
            cds[eid] = seq
    df = df[df["ensembl_id"].isin(cds.keys())].reset_index(drop=True)
    print(f"  with CDS on disk: {len(df)}")

    print("\n=== embed with NT-v2 ===")
    device = None if args.device == "auto" else args.device
    x_vecs = embed_all(cds, args.cache_dir, device=device)

    if "x" in df.columns:
        df = df.drop(columns=["x"])
    df["x"] = df["ensembl_id"].map(lambda e: x_vecs[e])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\n  wrote {out_path} ({len(df)} rows)")
    print("  per-family counts:")
    print(df["family"].value_counts().to_string())

    sample_vec = df["x"].iloc[0]
    print(f"  x shape per gene: {sample_vec.shape}  dtype: {sample_vec.dtype}")


if __name__ == "__main__":
    main()
