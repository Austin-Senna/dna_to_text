"""Extract self-supervised DNA encoder reductions over TSS-centered windows.

This is the TSS-context counterpart to ``scripts/run_multi_pool_extract.py``.
It reuses the same per-chunk mean/max/cls extraction machinery, but reads
196,608 bp TSS-centered genomic windows instead of CDS sequences.
"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd

from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, fetch_tss_window
from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.multi_pool import embed_all_multi_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def _load_encoder(name: str):
    spec = get_encoder_spec(name)
    module = import_module(spec.loader_module)
    return spec, module.load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nt_v2", choices=main_encoder_names())
    ap.add_argument("--template-dataset", default=str(DATA / "dataset_nt_v2_meanD.parquet"))
    ap.add_argument("--window-cache", default=str(DATA / "enformer_windows"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--length", type=int, default=ENFORMER_WINDOW_LENGTH)
    ap.add_argument("--max-genes", type=int, default=None, help="pilot limit; omit for full corpus")
    args = ap.parse_args()

    spec, load_fn = _load_encoder(args.encoder)
    cache_dir = Path(args.cache_dir) if args.cache_dir else DATA / f"tss_chunk_reductions_{spec.cache_name}"
    print(f"=== TSS {spec.display_name}: max_content_tokens={spec.max_content_tokens} stride={spec.stride} ===")
    print(f"  template: {Path(args.template_dataset).name}")
    print(f"  window cache: {args.window_cache}")
    print(f"  reduction cache: {cache_dir}")

    df = pd.read_parquet(args.template_dataset)
    if args.max_genes is not None:
        df = df.head(args.max_genes).copy()
        print(f"  pilot limit: {len(df)} genes")
    else:
        print(f"  genes: {len(df)}")

    windows: dict[str, str] = {}
    for eid in df["ensembl_id"]:
        seq = fetch_tss_window(eid, args.window_cache, length=args.length)
        if seq:
            windows[eid] = seq
    print(f"  with TSS windows: {len(windows)}")

    device = None if args.device == "auto" else args.device
    out = embed_all_multi_pool(
        windows,
        load_model_fn=load_fn,
        cache_dir=cache_dir,
        max_content_tokens=spec.max_content_tokens,
        stride=spec.stride,
        device=device,
        desc=f"tss {args.encoder} multi-pool",
    )
    print(f"  done: {len(out)} genes cached at {cache_dir}")


if __name__ == "__main__":
    main()
