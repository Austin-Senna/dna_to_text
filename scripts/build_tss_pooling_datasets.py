"""Materialize TSS-window pooling datasets from cached per-chunk reductions.

Reads ``EncoderSpec.tss_chunk_dir`` through ``multi_pool.load_reductions``, so
every gene's reductions must come from its current manifest window and the
pinned revision (G19). Writes the encoder's TSS grid
(``model_registry.encoder_pools(enc, "TSS")``) plus the TSS-only ``centermean``
template, and refuses a constant feature matrix (G3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.enformer_windows import read_window
from data_loader.model_registry import encoder_pools, get_encoder_spec, main_encoder_names
from data_loader.multi_pool import load_reductions
from data_loader.pooling_aggregator import TSS_POOLING_VARIANTS, aggregate, output_dim
from linear_trainer.sources import META_PARQUET, check_not_constant

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nt_v2", choices=main_encoder_names())
    ap.add_argument("--template-dataset", default=str(META_PARQUET))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--variants", nargs="+", default=None, choices=list(TSS_POOLING_VARIANTS),
                    help="default: the encoder's TSS grid plus centermean")
    args = ap.parse_args()

    spec = get_encoder_spec(args.encoder)
    allowed = (*encoder_pools(args.encoder, "TSS"), "centermean")
    variants = args.variants or list(allowed)
    stray = sorted(set(variants) - set(allowed))
    if stray:
        raise ValueError(f"{args.encoder} has no TSS {stray} pools")
    chunk_dir = Path(args.cache_dir) if args.cache_dir else spec.tss_chunk_dir
    if not chunk_dir.exists():
        raise FileNotFoundError(f"missing TSS chunk reductions for {args.encoder}: {chunk_dir}")

    base = pd.read_parquet(args.template_dataset)
    print(f"=== TSS {args.encoder}: {len(base)} genes from {Path(args.template_dataset).name} ===")
    print(f"  loading TSS chunk reductions from {chunk_dir}...")

    per_gene = load_reductions(chunk_dir, spec, {eid: read_window(eid) for eid in base["ensembl_id"]})
    sample_d = next(iter(per_gene.values()))["mean"].shape[1]
    print(f"  per-chunk dim d={sample_d}")

    for variant in variants:
        out_path = DATA / f"dataset_tss_{spec.dataset_stem}_{variant}.parquet"
        print(f"  building {variant} ({output_dim(variant, sample_d)} dim) -> {out_path.name}")
        x_col = [aggregate(per_gene[eid], variant) for eid in base["ensembl_id"]]
        assert all(v.shape == (output_dim(variant, sample_d),) for v in x_col), (
            f"variant {variant}: dim mismatch in some rows"
        )
        check_not_constant(np.stack(x_col), f"tss_{args.encoder}_{variant}")

        new_df = base.copy()
        new_df["x"] = x_col
        new_df.to_parquet(out_path)
        print(f"    wrote {len(new_df)} rows")
        if variant == "meanmean":
            alias = DATA / f"dataset_tss_{spec.dataset_stem}.parquet"
            new_df.to_parquet(alias)
            print(f"    wrote base alias -> {alias.name}")


if __name__ == "__main__":
    main()
