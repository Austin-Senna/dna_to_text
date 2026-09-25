"""Check that the local MMseqs2 build reproduces the canonical homology split.

Clusters the corpus into a scratch directory (``data/splits.json`` is never
touched), re-assigns clusters to partitions at the canonical seed, and compares
gene membership with the committed split. Different MMseqs2 builds can cluster
differently, so run this before any script that re-clusters
(``seed_sensitivity.py``) and record the printed version with the results.

Run: uv run scripts/check_split_reproduction.py [--mmseqs PATH] [--id 0.4]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from cluster import cluster_dataframe
from splits.loader import resolve_dataset_path
from splits.make_splits import write_cluster_splits_json

DATA = Path(__file__).resolve().parents[1] / "data"
CANONICAL = {0.4: DATA / "splits.json", 0.7: DATA / "splits_homology70.json"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 binary")
    ap.add_argument("--id", type=float, default=0.4, choices=sorted(CANONICAL))
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--keep-tsv", type=Path,
                    help="copy the cluster TSV here (e.g. data/clusters/homology_id40.tsv)")
    args = ap.parse_args()

    version = subprocess.run([args.mmseqs, "version"], capture_output=True, text=True).stdout.strip()
    canonical = json.loads(CANONICAL[args.id].read_text())
    df = pd.read_parquet(resolve_dataset_path(), columns=["ensembl_id", "family"]).drop_duplicates("ensembl_id")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        df_clustered, stats = cluster_dataframe(df, work, min_seq_id=args.id,
                                                coverage=args.coverage, mmseqs_bin=args.mmseqs)
        out = work / "splits.json"
        write_cluster_splits_json(df_clustered, out, stats, seed=canonical["seed"])
        rebuilt = json.loads(out.read_text())
        if args.keep_tsv:
            args.keep_tsv.parent.mkdir(parents=True, exist_ok=True)
            args.keep_tsv.write_bytes((work / "res_cluster.tsv").read_bytes())

    match = {k: sorted(canonical[k]) == sorted(rebuilt[k]) for k in ("train", "val", "test")}
    print(f"mmseqs {version}  id={args.id}  clusters={stats['n_clusters']} "
          f"(canonical {canonical.get('cluster_stats', {}).get('n_clusters', '?')})")
    print("partition match:", match)
    raise SystemExit(0 if all(match.values()) else 1)


if __name__ == "__main__":
    main()
