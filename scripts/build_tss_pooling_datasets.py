"""Materialize TSS-window pooling datasets from cached per-chunk reductions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.pooling_aggregator import (
    POOLING_VARIANTS,
    aggregate,
    available_variants,
    output_dim,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nt_v2", choices=main_encoder_names())
    ap.add_argument("--template-dataset", default=str(DATA / "dataset_nt_v2_meanD.parquet"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(POOLING_VARIANTS),
        choices=list(POOLING_VARIANTS),
    )
    args = ap.parse_args()

    spec = get_encoder_spec(args.encoder)
    chunk_dir = Path(args.cache_dir) if args.cache_dir else DATA / f"tss_chunk_reductions_{spec.cache_name}"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"missing TSS chunk reductions for {args.encoder}: {chunk_dir}")

    base = pd.read_parquet(args.template_dataset)
    print(f"=== TSS {args.encoder}: {len(base)} genes from {Path(args.template_dataset).name} ===")
    print(f"  loading TSS chunk reductions from {chunk_dir}...")

    per_gene: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for eid in base["ensembl_id"]:
        f = chunk_dir / f"{eid}.npz"
        if not f.exists():
            missing.append(eid)
            continue
        with np.load(f) as data:
            per_gene[eid] = {k: data[k] for k in ("mean", "max", "cls") if k in data.files}
    if missing:
        raise RuntimeError(
            f"TSS chunk reductions missing for {len(missing)} genes: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    sample_d = next(iter(per_gene.values()))["mean"].shape[1]
    supported = set(available_variants(next(iter(per_gene.values()))))
    print(f"  per-chunk dim d={sample_d}")

    for variant in args.variants:
        if variant not in supported:
            print(f"  skipping {variant}: not supported by cached reductions")
            continue

        out_path = DATA / f"dataset_tss_{spec.dataset_stem}_{variant}.parquet"
        print(f"  building {variant} ({output_dim(variant, sample_d)} dim) -> {out_path.name}")
        x_col = [aggregate(per_gene[eid], variant) for eid in base["ensembl_id"]]
        assert all(v.shape == (output_dim(variant, sample_d),) for v in x_col), (
            f"variant {variant}: dim mismatch in some rows"
        )

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
