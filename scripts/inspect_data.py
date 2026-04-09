"""Inspect every artifact produced by the pipeline.

Usage:
    python scripts/inspect.py             # inspect everything that exists
    python scripts/inspect.py hgnc        # only HGNC TSVs
    python scripts/inspect.py gene_table  # only gene_table.parquet
    python scripts/inspect.py sequences   # only CDS FASTAs
    python scripts/inspect.py dataset     # only dataset.parquet
    python scripts/inspect.py embeddings  # only cached .npy files
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def banner(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _read_fasta(p: Path) -> str:
    return "".join(l.strip() for l in p.read_text().splitlines() if not l.startswith(">"))


def inspect_hgnc():
    banner("HGNC TSVs (data/hgnc/)")
    d = DATA / "hgnc"
    if not d.exists():
        print("  (not downloaded yet — run prepare_data.py)")
        return
    for tsv in sorted(d.glob("*.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        sym_col = next((c for c in df.columns if "approved" in c.lower() and "symbol" in c.lower()), df.columns[0])
        print(f"  {tsv.name:20s} {len(df):5d} rows  e.g. {df[sym_col].head(3).tolist()}")


def inspect_gene_table():
    banner("gene_table.parquet")
    p = DATA / "gene_table.parquet"
    if not p.exists():
        print("  (not built yet — run prepare_data.py)")
        return
    df = pd.read_parquet(p)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print("\n  per-family counts:")
    print(df["family"].value_counts().to_string())
    row = df.iloc[0]
    print(f"\n  sample row [{row['symbol']} / {row['ensembl_id']} / {row['family']}]:")
    print(f"    summary[:200]   = {str(row['summary'])[:200]!r}")
    print(f"    y_embedding dim = {np.asarray(row['y_embedding']).shape}")


def inspect_sequences():
    banner("CDS FASTAs (data/sequences/)")
    d = DATA / "sequences"
    if not d.exists():
        print("  (not fetched yet — run prepare_data.py)")
        return
    files = sorted(d.glob("*.fa"))
    if not files:
        print("  (empty)")
        return
    lens = [len(_read_fasta(f)) for f in files]
    print(f"  count : {len(files)}")
    print(f"  median: {int(np.median(lens))} nt")
    print(f"  min   : {min(lens)} nt")
    print(f"  max   : {max(lens)} nt")
    print(f"\n  sample [{files[0].name}]:")
    print(f"    {_read_fasta(files[0])[:200]}...")


def inspect_embeddings():
    banner("DNABERT-2 embeddings (data/embeddings/)")
    d = DATA / "embeddings"
    if not d.exists():
        print("  (not embedded yet — run run_encoder.py)")
        return
    files = sorted(d.glob("*.npy"))
    if not files:
        print("  (empty)")
        return
    arr = np.load(files[0])
    print(f"  count    : {len(files)}")
    print(f"  shape    : {arr.shape}")
    print(f"  dtype    : {arr.dtype}")
    print(f"  sample[:5]: {arr[:5]}")


def inspect_dataset():
    banner("dataset.parquet (final)")
    p = DATA / "dataset.parquet"
    if not p.exists():
        print("  (not built yet — run run_encoder.py)")
        return
    df = pd.read_parquet(p)
    print(f"  shape  : {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print("\n  per-family counts:")
    print(df["family"].value_counts().to_string())
    row = df.iloc[0]
    x = np.asarray(row["x"])
    y = np.asarray(row["y"])
    print(f"\n  sample row [{row['symbol']}]:")
    print(f"    x (DNABERT-2): shape={x.shape}  norm={np.linalg.norm(x):.3f}")
    print(f"    y (GenePT)   : shape={y.shape}  norm={np.linalg.norm(y):.3f}")
    print(f"    NaN x: {np.isnan(x).any()}   NaN y: {np.isnan(y).any()}")


STEPS = {
    "hgnc": inspect_hgnc,
    "gene_table": inspect_gene_table,
    "sequences": inspect_sequences,
    "embeddings": inspect_embeddings,
    "dataset": inspect_dataset,
}


def main():
    args = sys.argv[1:]
    if not args:
        for fn in STEPS.values():
            fn()
        return
    for a in args:
        if a not in STEPS:
            print(f"unknown: {a}. choices: {list(STEPS)}")
            sys.exit(1)
        STEPS[a]()


if __name__ == "__main__":
    main()
