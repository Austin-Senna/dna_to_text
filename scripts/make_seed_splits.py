"""Write the split-seed robustness splits, each to its own file (G15).

Two kinds per seed, both re-assigned from tracked inputs with no MMseqs2 rerun:
  * CDS: ``data/splits_seed{N}.json``, whole 40%-identity protein clusters
    (``data/clusters/homology_id40.tsv``) assigned by the same greedy assigner
    as ``data/splits.json``. With seed 42 this reproduces ``splits.json``
    exactly (a test checks it), so the seeds differ from the primary split
    only in the seed.
  * TSS: ``data/splits_tss_disjoint_seed{N}.json``, the window-and-protein
    disjoint split (``make_tss_disjoint_split.build``) at seed N, the TSS
    primary's robustness splits (decided Sept 29).

The CDS assigner visits clusters in the families parquet's row order, so that
order is part of the input: the payload stamps a hash of it, and a rebuild
from a parquet in another order raises instead of moving genes.

The primary splits are never written here: seed 42 is refused.

Run: uv run scripts/make_seed_splits.py [--seeds 1 7 123] [--check]
  --check rebuilds every file in memory and fails if any tracked file differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from cluster.mmseqs_cluster import parse_cluster_tsv
from splits.make_splits import (DEFAULT_FRACS, SEED, STRATIFY_COL, build_cluster_splits,
                                family_proportions)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_tss_disjoint_split as mtsd  # noqa: E402

SEEDS = (1, 7, 123)
CLUSTER_TSV = DATA / "clusters" / "homology_id40.tsv"
FAMILIES = DATA / "dataset_dnabert2_meanmean.parquet"
UNIVERSE = DATA / "splits.json"
# Hash of the families parquet's gene order, which the greedy assigner depends on.
ROW_ORDER_SHA256 = "abc918eb1bc02224df8298a8097252c2a32a8a019b87e05aff3edc167c839db1"


class RowOrderChanged(RuntimeError):
    """The families parquet lists the genes in a different order than the split was built from."""


class StaleSeedSplit(RuntimeError):
    """A tracked seed split differs from its rebuild."""


def row_order_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def cds_payload(seed: int, families: Path = FAMILIES, clusters: Path = CLUSTER_TSV,
                expect_order: str | None = ROW_ORDER_SHA256) -> dict:
    """The CDS homology split at ``seed``, from the tracked 40% clusters."""
    df = pd.read_parquet(families, columns=["ensembl_id", STRATIFY_COL]).drop_duplicates("ensembl_id")
    order = row_order_sha256(df["ensembl_id"].tolist())
    if expect_order is not None and order != expect_order:
        raise RowOrderChanged(f"{families}: gene order hash {order[:12]}, expected {expect_order[:12]}")
    pmap = parse_cluster_tsv(clusters)
    missing = sorted(set(df["ensembl_id"]) - set(pmap))
    if missing:
        raise KeyError(f"{len(missing)} genes have no protein cluster: {missing[:5]}")
    df = df.assign(cluster_id=df["ensembl_id"].map(pmap))
    parts = build_cluster_splits(df, seed=seed)
    return {
        **parts,
        "seed": seed,
        "stratify": STRATIFY_COL,
        "method": "homology_cluster",
        "fracs": list(DEFAULT_FRACS),
        "cluster_stats": {"n_genes": len(df), "n_clusters": int(df["cluster_id"].nunique()),
                          "min_seq_id": 0.40, "coverage": 0.80},
        "family_proportions": family_proportions(df, parts),
        "inputs": {"cluster_tsv": mtsd._stamp(clusters),
                   "families": {**mtsd._labels_stamp(families, df), "row_order_sha256": order}},
    }


def tss_payload(seed: int) -> dict:
    out = DATA / f"splits_tss_disjoint_seed{seed}.json"
    payload, _, _ = mtsd.build(mtsd.MANIFEST, mtsd.CLUSTER_TSV, mtsd.FAMILIES, UNIVERSE, seed, out.name)
    return payload


def targets(seed: int) -> dict[Path, dict]:
    if seed == SEED:
        raise ValueError(f"seed {SEED} is the primary split; it is built by make_splits.py "
                         "and make_tss_disjoint_split.py")
    return {DATA / f"splits_seed{seed}.json": cds_payload(seed),
            DATA / f"splits_tss_disjoint_seed{seed}.json": tss_payload(seed)}


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and fail if a tracked file differs")
    args = ap.parse_args()
    stale = []
    for seed in args.seeds:
        for path, payload in targets(seed).items():
            text = render(payload)
            rel = path.relative_to(REPO_ROOT)
            if args.check:
                if not path.exists() or path.read_text() != text:
                    stale.append(str(rel))
                continue
            path.write_text(text)
            print(f"wrote {rel}: train={len(payload['train'])} val={len(payload['val'])} "
                  f"test={len(payload['test'])}")
    if stale:
        raise StaleSeedSplit(f"seed splits differ from a rebuild: {stale}")
    if args.check:
        print("every seed split matches its rebuild")


if __name__ == "__main__":
    main()
