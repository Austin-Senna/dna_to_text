"""Phase 4b: materialise per-pooling-variant dataset parquets from cached
per-chunk reductions.

For each (encoder, variant) pair, builds a parquet with the same schema as
dataset.parquet / dataset_nt_v2.parquet — `symbol, ensembl_id, family,
summary, y, x` — but `x` is the variant's reduction.

Reads chunk reductions from `data/chunk_reductions_{encoder}/{ENSG}.npz`.
Reads y (GenePT) from the existing dataset parquet for that encoder so
we don't duplicate the embedding-loading code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.pooling_aggregator import aggregate, output_dim, POOLING_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

ENCODER_BASE_DATASETS = {
    "dnabert2": DATA / "dataset.parquet",
    "nt_v2":    DATA / "dataset_nt_v2.parquet",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODER_BASE_DATASETS.keys()))
    ap.add_argument(
        "--variants", nargs="+", default=list(POOLING_VARIANTS),
        choices=list(POOLING_VARIANTS),
    )
    args = ap.parse_args()

    chunk_dir = DATA / f"chunk_reductions_{args.encoder}"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"missing chunk reductions for {args.encoder}: {chunk_dir}")

    base = pd.read_parquet(ENCODER_BASE_DATASETS[args.encoder])
    print(f"=== {args.encoder}: {len(base)} genes from {ENCODER_BASE_DATASETS[args.encoder].name} ===")

    # Pre-load all per-chunk reductions so we don't reread per variant.
    print(f"  loading chunk reductions from {chunk_dir}...")
    per_gene: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for eid in base["ensembl_id"]:
        f = chunk_dir / f"{eid}.npz"
        if not f.exists():
            missing.append(eid)
            continue
        with np.load(f) as data:
            per_gene[eid] = {k: data[k] for k in ("mean", "max", "cls")}
    if missing:
        raise RuntimeError(
            f"chunk reductions missing for {len(missing)} genes: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    sample_d = next(iter(per_gene.values()))["mean"].shape[1]
    print(f"  per-chunk dim d={sample_d}")

    for variant in args.variants:
        out_path = DATA / f"dataset_{args.encoder}_{variant}.parquet"
        print(f"  building {variant} ({output_dim(variant, sample_d)} dim) -> {out_path.name}")

        x_col = []
        for eid in base["ensembl_id"]:
            x_col.append(aggregate(per_gene[eid], variant))

        new_df = base.copy()
        new_df["x"] = x_col
        # Sanity: every row has expected dim
        assert all(v.shape == (output_dim(variant, sample_d),) for v in x_col), \
            f"variant {variant}: dim mismatch in some rows"
        new_df.to_parquet(out_path)
        print(f"    wrote {len(new_df)} rows")


if __name__ == "__main__":
    main()
