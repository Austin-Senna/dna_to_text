"""Assemble an ESM-2 feature dataset parquet (#9).

Takes the shared {ensembl_id, symbol, family, summary, y} metadata + GenePT target
from an existing dataset parquet and sets `x` to the cached ESM-2 embedding, so the
result is probe-compatible with every other dataset_*.parquet.

Run:
  uv run scripts/build_esm2_datasets.py --size 150m
  uv run scripts/build_esm2_datasets.py --size 650m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader.cache_meta import StaleCache, read_meta, sha256_text  # noqa: E402
from data_loader.sequence_fetcher import fetch_cds  # noqa: E402
from protein import translate_cds  # noqa: E402
from splits.loader import resolve_dataset_path  # noqa: E402

DATA = REPO_ROOT / "data"
DIMS = {"150m": 640, "650m": 1280}
MODELS = {"150m": "esm2_t30_150M_UR50D", "650m": "esm2_t33_650M_UR50D"}
RUN_KEYS = ("model", "fp16", "device", "max_residues", "translation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(DIMS), required=True)
    ap.add_argument("--template", type=Path, default=None,
                    help="dataset parquet supplying meta+y (default: auto-resolve)")
    ap.add_argument("--emb-cache", type=Path, default=None)
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    emb_cache = args.emb_cache or (DATA / f"esm2_{args.size}_embeddings_v2")
    out = args.out or (DATA / f"dataset_esm2_{args.size}.parquet")
    template = args.template or resolve_dataset_path()

    base = pd.read_parquet(template)
    keep = [c for c in ("ensembl_id", "symbol", "family", "summary", "y") if c in base.columns]
    base = base[keep].copy()
    print(f"=== ESM-2 {args.size} dataset: {len(base)} genes from {Path(template).name} ===")

    xs, missing, runs = [], [], set()
    for eid in base["ensembl_id"]:
        f = emb_cache / f"{eid}.npz"
        if not f.exists():
            missing.append(eid)
            continue
        rec = read_meta(f)
        protein = translate_cds(fetch_cds(eid, args.seq_cache), mode="through")
        if rec is None or rec["protein_sha256"] != sha256_text(protein):
            raise StaleCache(f"{f}: embedding was not built from the current protein")
        runs.add(tuple(rec[k] for k in RUN_KEYS))
        with np.load(f, allow_pickle=False) as data:
            xs.append(data["emb"].astype(np.float32))
    if missing:
        raise RuntimeError(f"{len(missing)} genes have no embedding in {emb_cache}: "
                         f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if len(runs) != 1 or next(iter(runs))[0] != MODELS[args.size]:
        raise StaleCache(f"embeddings come from more than one run or the wrong model: {sorted(runs)}")
    base["x"] = xs

    dims = {int(v.shape[0]) for v in base["x"]}
    assert dims == {DIMS[args.size]}, f"unexpected embedding dims {dims}, expected {DIMS[args.size]}"
    base.to_parquet(out)
    print(f"  wrote {len(base)} rows (x dim {DIMS[args.size]}) -> {out.name}")


if __name__ == "__main__":
    main()
