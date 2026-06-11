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

from splits.loader import resolve_dataset_path  # noqa: E402

DATA = REPO_ROOT / "data"
DIMS = {"150m": 640, "650m": 1280}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(DIMS), required=True)
    ap.add_argument("--template", type=Path, default=None,
                    help="dataset parquet supplying meta+y (default: auto-resolve)")
    ap.add_argument("--emb-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    emb_cache = args.emb_cache or (DATA / f"esm2_{args.size}_embeddings")
    out = args.out or (DATA / f"dataset_esm2_{args.size}.parquet")
    template = args.template or resolve_dataset_path()

    base = pd.read_parquet(template)
    keep = [c for c in ("ensembl_id", "symbol", "family", "summary", "y") if c in base.columns]
    base = base[keep].copy()
    print(f"=== ESM-2 {args.size} dataset: {len(base)} genes from {Path(template).name} ===")

    xs, missing = [], []
    for eid in base["ensembl_id"]:
        f = emb_cache / f"{eid}.npy"
        if not f.exists():
            missing.append(eid)
            xs.append(None)
            continue
        xs.append(np.load(f).astype(np.float32))
    if missing:
        print(f"  WARNING: {len(missing)} genes have no embedding (dropped): "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    base["x"] = xs
    base = base[base["x"].notna()].reset_index(drop=True)

    dims = {int(v.shape[0]) for v in base["x"]}
    assert dims == {DIMS[args.size]}, f"unexpected embedding dims {dims}, expected {DIMS[args.size]}"
    base.to_parquet(out)
    print(f"  wrote {len(base)} rows (x dim {DIMS[args.size]}) -> {out.name}")


if __name__ == "__main__":
    main()
