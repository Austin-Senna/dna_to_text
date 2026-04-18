"""CLI: build data/splits.json from data/dataset.parquet (stratified 70/15/15 by family)."""
from __future__ import annotations

import argparse
from pathlib import Path

from splits.make_splits import SEED, write_splits_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--out", default=str(DATA / "splits.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    payload = write_splits_json(args.dataset, args.out, seed=args.seed)
    print(f"wrote {args.out}")
    print(f"  train: {len(payload['train'])}")
    print(f"  val:   {len(payload['val'])}")
    print(f"  test:  {len(payload['test'])}")
    print(f"  seed:  {payload['seed']}  stratify={payload['stratify']}")


if __name__ == "__main__":
    main()
