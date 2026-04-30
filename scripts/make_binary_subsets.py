"""Build the two frozen binary subsets and write them to data/."""
from __future__ import annotations

import argparse
from pathlib import Path

from binary_tasks import write_binary_subset_json, BINARY_TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for task in BINARY_TASKS:
        out = out_dir / f"binary_{task}.json"
        payload = write_binary_subset_json(task, out, dataset_path=args.dataset)
        n = payload["n_per_class"]
        sp = payload["split"]
        print(
            f"  {task}: n_per_class={n}  "
            f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}  "
            f"-> {out}"
        )


if __name__ == "__main__":
    main()
