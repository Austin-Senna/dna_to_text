"""Build a genomic-interval-disjoint homology split for the TSS arm (MLCB R2-Q4).

The primary homology split (``data/splits.json``) clusters translated proteins at
40% identity, but the 196,608 bp TSS-centered windows of genomically adjacent genes
that land in *different* protein clusters still overlap (48.3% of test genes share a
window with a train/val gene). This builds a stricter split that is disjoint on BOTH
axes: genes are grouped by the union of two edge sets — (a) overlapping TSS windows,
(b) shared 40%-identity protein cluster — and whole groups are assigned family-balanced
70/15/15 by the same greedy assigner the primary split uses. No TSS window can then
straddle the train/val/test boundary.

Reuses the cached MMseqs2 cluster tsv (no re-clustering) and the loader's own
``centered_window`` so the window geometry is identical to what the encoders saw.

Run: uv run scripts/make_tss_disjoint_split.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from cluster.mmseqs_cluster import parse_cluster_tsv
from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, centered_window
from splits.loader import resolve_dataset_path
from splits.make_splits import SEED, write_cluster_splits_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
LOOKUP_DIR = DATA / "enformer_windows" / "_lookup"
CLUSTER_TSV = DATA / "cluster_work" / "id40" / "res_cluster.tsv"


class UnionFind:
    """Minimal union-find over gene ids (path-halving + union-by-nothing)."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _windows(genes: list[str]) -> list[tuple[str, str, int, int]]:
    """Reconstruct each gene's (chrom, w0, w1) TSS window from the cached lookups."""
    out: list[tuple[str, str, int, int]] = []
    missing: list[str] = []
    for g in genes:
        lk = LOOKUP_DIR / f"{g}.json"
        if not lk.exists():
            missing.append(g)
            continue
        d = json.loads(lk.read_text())
        chrom, w0, w1 = centered_window(
            seq_region_name=str(d["seq_region_name"]),
            start=int(d["start"]),
            end=int(d["end"]),
            strand=int(d.get("strand", 1)),
        )
        out.append((g, chrom, w0, w1))
    if missing:
        raise SystemExit(f"{len(missing)} genes missing window lookup: {missing[:10]}")
    return out


def _window_overlap_edges(windows: list[tuple[str, str, int, int]]) -> list[tuple[str, str]]:
    """Every pair of genes whose windows overlap on the same chromosome (sweep)."""
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for g, chrom, w0, w1 in windows:
        by_chrom[chrom].append((w0, w1, g))
    edges: list[tuple[str, str]] = []
    for recs in by_chrom.values():
        recs.sort()
        n = len(recs)
        for i in range(n):
            w0i, w1i, gi = recs[i]
            for j in range(i + 1, n):
                w0j, _w1j, gj = recs[j]
                if w0j > w1i:  # sorted by w0: nothing further can overlap i
                    break
                edges.append((gi, gj))
    return edges


def _cross_split_overlaps(
    windows: list[tuple[str, str, int, int]],
    split_of: dict[str, str],
) -> int:
    """Count gene pairs in different splits whose windows overlap (must be 0)."""
    return sum(
        1
        for gi, gj in _window_overlap_edges(windows)
        if split_of[gi] != split_of[gj]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default=None,
                    help="parquet for gene/family table (default: auto-resolve)")
    ap.add_argument("--out", type=Path, default=DATA / "splits_tss_disjoint.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    dataset_path = resolve_dataset_path(Path(args.dataset) if args.dataset else None)
    df = pd.read_parquet(dataset_path, columns=["ensembl_id", "family"])
    df = df.drop_duplicates("ensembl_id").reset_index(drop=True)
    genes = df["ensembl_id"].tolist()
    print(f"=== gene table from {dataset_path.name}: {len(genes)} genes ===")

    windows = _windows(genes)
    win_edges = _window_overlap_edges(windows)
    protein_map = parse_cluster_tsv(CLUSTER_TSV)  # member -> representative
    print(f"  window-overlap edges: {len(win_edges)}; "
          f"protein clusters: {len(set(protein_map.values()))}")

    uf = UnionFind(genes)
    for gi, gj in win_edges:
        uf.union(gi, gj)
    for member, rep in protein_map.items():
        if member in uf.parent and rep in uf.parent:
            uf.union(member, rep)

    df["cluster_id"] = df["ensembl_id"].map(uf.find)
    n_groups = df["cluster_id"].nunique()
    print(f"  combined groups (window-overlap ∪ protein-cluster): {n_groups}")

    stats = {
        "method": "genomic_window_and_protein_cluster_disjoint",
        "n_genes": len(genes),
        "n_window_overlap_edges": len(win_edges),
        "n_protein_clusters": len(set(protein_map.values())),
        "n_combined_groups": int(n_groups),
        "window_length_bp": ENFORMER_WINDOW_LENGTH,
        "protein_min_seq_id": 0.40,
        "protein_coverage": 0.80,
    }
    payload = write_cluster_splits_json(df, args.out, stats, seed=args.seed)

    split_of = {g: s for s in ("train", "val", "test") for g in payload[s]}
    residual = _cross_split_overlaps(windows, split_of)
    print(f"\nwrote {args.out.name}: train={len(payload['train'])} "
          f"val={len(payload['val'])} test={len(payload['test'])}")
    for split, props in payload["family_proportions"].items():
        print(f"    {split:<5s} {props}")
    print(f"\nresidual cross-split window overlaps: {residual} (must be 0)")
    if residual != 0:
        raise SystemExit(f"FAILED: {residual} cross-split window overlaps remain")
    print("OK: windows are interval-disjoint across splits.")


if __name__ == "__main__":
    main()
