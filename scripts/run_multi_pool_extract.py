"""Phase 4b: re-extract per-chunk reductions (mean / max / cls) for one encoder.

One forward pass per chunk; stores three per-chunk arrays per gene to
`data/chunk_reductions_{encoder}/{ENSG}.npz`. Idempotent: cached genes
are skipped on rerun. Materialise the per-pooling-variant datasets with
`scripts/build_pooling_datasets.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from data_loader.multi_pool import embed_all_multi_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


# Encoder-specific load functions and chunking. Content tokens = max - 2 to
# leave room for CLS and SEP wrapped per chunk inside multi_pool.embed_sequence.
def _load_dnabert2():
    from data_loader.encoder_runner import load_model
    return load_model, 510, 64  # 512 - 2 special tokens

def _load_nt_v2():
    from data_loader.nt_v2_encoder import load_model
    return load_model, 998, 64  # 1000 - 2 special tokens


ENCODERS = {
    "dnabert2": _load_dnabert2,
    "nt_v2":    _load_nt_v2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS.keys()))
    ap.add_argument("--gene-table", default=str(DATA / "gene_table.parquet"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    args = ap.parse_args()

    load_fn, max_tokens, stride = ENCODERS[args.encoder]()
    cache_dir = DATA / f"chunk_reductions_{args.encoder}"
    print(f"=== {args.encoder}: max_content_tokens={max_tokens} stride={stride} ===")
    print(f"  cache dir: {cache_dir}")

    df = pd.read_parquet(args.gene_table)
    print(f"  gene table: {len(df)} genes")

    cds: dict[str, str] = {}
    for eid in df["ensembl_id"]:
        seq = fetch_cds(eid, DATA / "sequences")
        if seq:
            cds[eid] = seq
    print(f"  with CDS on disk: {len(cds)}")

    device = None if args.device == "auto" else args.device
    out = embed_all_multi_pool(
        cds,
        load_model_fn=load_fn,
        cache_dir=cache_dir,
        max_content_tokens=max_tokens,
        stride=stride,
        device=device,
        desc=f"{args.encoder} multi-pool",
    )
    print(f"  done: {len(out)} genes cached at {cache_dir}")


if __name__ == "__main__":
    main()
