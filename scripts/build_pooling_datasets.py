"""Phase 4b: materialise per-pooling-variant dataset parquets from cached
per-chunk reductions.

For each (encoder, variant) pair, builds a parquet with the same schema as
dataset.parquet / dataset_nt_v2.parquet — `symbol, ensembl_id, family,
summary, y, x` — but `x` is the variant's reduction.

Reads chunk reductions from ``EncoderSpec.chunk_dir`` through
``multi_pool.load_reductions``, which checks each gene against its current CDS
and the pinned revision (G19). Writes only the encoder's own pools
(``EncoderSpec.pooling_variants``; HyenaDNA has no clsmean/specialmean) and
refuses a constant feature matrix (G3).
Reads y (GenePT) from the existing dataset parquet for that encoder so
we don't duplicate the embedding-loading code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.model_registry import (
    BASE_DATASET_ALIASES,
    get_encoder_spec,
    main_encoder_names,
)
from data_loader.multi_pool import load_reductions
from data_loader.pooling_aggregator import (
    POOLING_VARIANTS,
    aggregate,
    output_dim,
)
from data_loader.sequence_fetcher import fetch_cds
from linear_trainer.sources import META_PARQUET, check_not_constant

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=main_encoder_names())
    ap.add_argument("--template-dataset", default=str(META_PARQUET))
    ap.add_argument("--variants", nargs="+", default=None, choices=list(POOLING_VARIANTS),
                    help="default: the encoder's own pools")
    args = ap.parse_args()

    spec = get_encoder_spec(args.encoder)
    variants = args.variants or list(spec.pooling_variants)
    stray = sorted(set(variants) - set(spec.pooling_variants))
    if stray:
        raise ValueError(f"{args.encoder} has no {stray} pools (EncoderSpec.pooling_variants)")
    chunk_dir = spec.chunk_dir
    if not chunk_dir.exists():
        raise FileNotFoundError(f"missing chunk reductions for {args.encoder}: {chunk_dir}")

    base = pd.read_parquet(args.template_dataset)
    print(f"=== {args.encoder}: {len(base)} genes from {Path(args.template_dataset).name} ===")

    # Pre-load all per-chunk reductions so we don't reread per variant.
    print(f"  loading chunk reductions from {chunk_dir}...")
    per_gene = load_reductions(chunk_dir, spec, {eid: fetch_cds(eid, DATA / "sequences")
                                                 for eid in base["ensembl_id"]})
    sample_d = next(iter(per_gene.values()))["mean"].shape[1]
    print(f"  per-chunk dim d={sample_d}")

    for variant in variants:
        out_path = spec.variant_dataset_path(variant)
        print(f"  building {variant} ({output_dim(variant, sample_d)} dim) -> {out_path.name}")

        x_col = []
        for eid in base["ensembl_id"]:
            x_col.append(aggregate(per_gene[eid], variant))

        new_df = base.copy()
        new_df["x"] = x_col
        # Sanity: every row has expected dim
        assert all(v.shape == (output_dim(variant, sample_d),) for v in x_col), \
            f"variant {variant}: dim mismatch in some rows"
        check_not_constant(np.stack(x_col), f"{args.encoder}_{variant}")
        new_df.to_parquet(out_path)
        print(f"    wrote {len(new_df)} rows")
        if variant == "meanmean" and args.encoder not in BASE_DATASET_ALIASES:
            new_df.to_parquet(spec.base_dataset_path)
            print(f"    wrote base alias -> {spec.base_dataset_path.name}")


if __name__ == "__main__":
    main()
