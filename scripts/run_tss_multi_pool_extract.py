"""Extract self-supervised DNA encoder reductions over TSS-centered windows.

This is the TSS-context counterpart to ``scripts/run_multi_pool_extract.py``.
It reuses the same per-chunk extraction machinery, but reads the 196,608 bp
canonical-TSS windows (gene orientation, TSS at index 98,304) through
``enformer_windows.read_window``, which checks each against the manifest.
"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from data_loader.enformer_windows import load_manifest, read_window
from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.multi_pool import embed_all_multi_pool


def _load_encoder(name: str):
    spec = get_encoder_spec(name)
    module = import_module(spec.loader_module)
    return spec, module.load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nt_v2", choices=main_encoder_names())
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--max-genes", type=int, default=None, help="pilot limit; omit for full corpus")
    args = ap.parse_args()

    spec, load_fn = _load_encoder(args.encoder)
    cache_dir = Path(args.cache_dir) if args.cache_dir else spec.tss_chunk_dir
    print(f"=== TSS {spec.display_name}: max_content_tokens={spec.max_content_tokens} stride={spec.stride} ===")
    print(f"  reduction cache: {cache_dir}")

    genes = sorted(load_manifest().index)[:args.max_genes]
    print(f"  genes: {len(genes)}" + (" (pilot limit)" if args.max_genes else ""))
    windows = {eid: read_window(eid) for eid in genes}

    device = None if args.device == "auto" else args.device
    out = embed_all_multi_pool(
        windows,
        load_model_fn=load_fn,
        cache_dir=cache_dir,
        spec=spec,
        device=device,
        desc=f"tss {args.encoder} multi-pool",
        collect=False,
    )
    print(f"  done: {len(out)} genes cached at {cache_dir}")


if __name__ == "__main__":
    main()
