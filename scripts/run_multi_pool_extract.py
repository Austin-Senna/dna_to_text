"""Extract per-chunk reductions over each gene's CDS for one encoder.

One forward pass per chunk; stores the per-chunk arrays per gene, with a meta
record, in ``EncoderSpec.chunk_dir`` (``data/chunk_reductions_v2_{encoder}``) or
``--cache-dir``.
A rerun reuses a gene only if its meta matches (same CDS, revision, chunking,
boundary tokens; G19). Materialise the pooled datasets with
`scripts/build_pooling_datasets.py`.
"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd

from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.sequence_fetcher import fetch_cds
from data_loader.multi_pool import embed_all_multi_pool
from linear_trainer.sources import META_PARQUET

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def _load_encoder(name: str):
    spec = get_encoder_spec(name)
    module = import_module(spec.loader_module)
    return spec, module.load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=main_encoder_names())
    ap.add_argument("--gene-table", default=str(META_PARQUET))
    ap.add_argument("--cache-dir", default=None, help="default: the encoder's chunk_dir")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    args = ap.parse_args()

    spec, load_fn = _load_encoder(args.encoder)
    cache_dir = Path(args.cache_dir) if args.cache_dir else spec.chunk_dir
    print(f"=== {spec.display_name}: max_content_tokens={spec.max_content_tokens} stride={spec.stride} ===")
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
        spec=spec,
        device=device,
        desc=f"{args.encoder} multi-pool",
    )
    print(f"  done: {len(out)} genes cached at {cache_dir}")


if __name__ == "__main__":
    main()
