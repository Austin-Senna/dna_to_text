"""CLI: build train/val/test splits.

Default behaviour (journal revision): write three split files —
  * ``splits.json``            — PRIMARY, homology-aware MMseqs2 clusters @40% id
  * ``splits_homology70.json`` — stricter homology split @70% id (supplementary)
  * ``splits_random.json``     — the original random family-stratified split
                                  (kept as a sensitivity analysis)

Gene set + family labels come from any present encoder parquet (resolved via
``splits.loader.resolve_dataset_path``); CDS for translation come from the
cached FASTAs in ``data/sequences``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cluster import cluster_dataframe
from splits.loader import resolve_dataset_path
from splits.make_splits import SEED, write_cluster_splits_json, write_splits_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def _gene_table(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path, columns=["ensembl_id", "family"])
    return df.drop_duplicates("ensembl_id").reset_index(drop=True)


def _report(name: str, payload: dict) -> None:
    print(f"wrote {name}: train={len(payload['train'])} "
          f"val={len(payload['val'])} test={len(payload['test'])}")
    if "family_proportions" in payload:
        for split, props in payload["family_proportions"].items():
            print(f"    {split:<5s} {props}")
    if "cluster_stats" in payload:
        cs = payload["cluster_stats"]
        print(f"    clusters={cs['n_clusters']} from {cs['n_genes']} genes "
              f"(missing CDS: {cs['n_missing_cds']}) @id>={cs['min_seq_id']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="parquet for gene/family table (default: auto-resolve)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--primary-id", type=float, default=0.40,
                    help="MMseqs2 min-seq-id for the PRIMARY split (default 0.40)")
    ap.add_argument("--strict-id", type=float, default=0.70,
                    help="MMseqs2 min-seq-id for the supplementary split (default 0.70)")
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--workdir", default=str(DATA / "cluster_work"))
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()

    dataset_path = resolve_dataset_path(
        Path(args.dataset) if args.dataset else None
    )
    print(f"=== gene table from {dataset_path.name} ===")
    df = _gene_table(dataset_path)
    print(f"  {len(df)} genes, families: {sorted(df['family'].unique())}")

    # Random split (sensitivity) — reuse the original stratified splitter.
    if not args.skip_random:
        rand = write_splits_json(dataset_path, DATA / "splits_random.json", seed=args.seed)
        _report("splits_random.json", rand)

    # Primary homology split @ primary-id.
    print(f"\n=== MMseqs2 clustering @ id>={args.primary_id} (PRIMARY) ===")
    df_p, stats_p = cluster_dataframe(
        df, Path(args.workdir) / f"id{int(args.primary_id*100)}",
        min_seq_id=args.primary_id, coverage=args.coverage,
    )
    primary = write_cluster_splits_json(
        df_p, DATA / "splits.json", stats_p, seed=args.seed,
    )
    _report("splits.json (PRIMARY)", primary)

    # Stricter homology split @ strict-id (supplementary).
    print(f"\n=== MMseqs2 clustering @ id>={args.strict_id} (supplementary) ===")
    df_s, stats_s = cluster_dataframe(
        df, Path(args.workdir) / f"id{int(args.strict_id*100)}",
        min_seq_id=args.strict_id, coverage=args.coverage,
    )
    strict = write_cluster_splits_json(
        df_s, DATA / "splits_homology70.json", stats_s, seed=args.seed,
    )
    _report("splits_homology70.json", strict)


if __name__ == "__main__":
    main()
