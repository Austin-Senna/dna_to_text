"""The TSS-primary split: disjoint on protein homology AND on genomic windows.

Genes are grouped by the union of two edge sets: TSS windows that overlap on
the chromosome, and a shared 40%-identity protein cluster. Whole groups are
assigned family-balanced 70/15/15 by the same greedy assigner as the homology
split, so no window and no protein cluster straddles train/val/test.

The result depends only on the gene set, the spans, the clusters and the seed:
genes are sorted and each group is named by its smallest gene ID, so input row
order cannot move a gene between splits (G21).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from splits.make_splits import (
    DEFAULT_FRACS,
    SEED,
    SPLIT_NAMES,
    STRATIFY_COL,
    build_cluster_splits,
    family_proportions,
)
from splits.window_leak import Span, overlap_pairs, window_leak_stats


class _UnionFind:
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


def build_tss_disjoint(families: pd.DataFrame, spans: dict[str, Span],
                       protein_map: dict[str, str], seed: int = SEED,
                       fracs: tuple[float, float, float] = DEFAULT_FRACS) -> tuple[dict, dict]:
    """Assign window-and-protein groups to splits.

    ``families`` has ``ensembl_id`` and ``family``; ``spans`` maps each gene to
    its window's genomic span; ``protein_map`` maps cluster member ->
    representative and must cover every gene. Returns ``(parts, stats)``, where
    ``parts`` holds sorted ``train``/``val``/``test`` lists and the per-split
    ``family_proportions``.
    """
    df = (families[["ensembl_id", STRATIFY_COL]].drop_duplicates("ensembl_id")
          .sort_values("ensembl_id").reset_index(drop=True))
    genes = df["ensembl_id"].tolist()
    for what, have in (("window span", spans), ("protein cluster", protein_map)):
        missing = [g for g in genes if g not in have]
        if missing:
            raise KeyError(f"{len(missing)} genes have no {what}: {missing[:5]}")

    edges = overlap_pairs({g: spans[g] for g in genes})
    uf = _UnionFind(genes)
    for a, b, _ in edges:
        uf.union(a, b)
    # Join members by their representative's label, so a cluster stays whole
    # even when its representative is outside this gene set.
    gene_set = set(genes)
    first_of_rep: dict[str, str] = {}
    for member, rep in protein_map.items():
        if member in gene_set:
            uf.union(member, first_of_rep.setdefault(rep, member))
    members: dict[str, list[str]] = defaultdict(list)
    for g in genes:
        members[uf.find(g)].append(g)
    group_of = {g: min(ms) for ms in members.values() for g in ms}
    df["cluster_id"] = df["ensembl_id"].map(group_of)

    parts = build_cluster_splits(df, seed=seed, fracs=fracs)
    parts = {s: sorted(parts[s]) for s in SPLIT_NAMES}
    split_of = {g: s for s in SPLIT_NAMES for g in parts[s]}
    straddling = {}
    for member, rep in protein_map.items():
        if member in split_of:
            straddling.setdefault(rep, set()).add(split_of[member])
    straddling = sorted(r for r, where in straddling.items() if len(where) > 1)
    if straddling:
        raise RuntimeError(f"{len(straddling)} protein clusters straddle splits: {straddling[:5]}")
    residual = window_leak_stats(parts, {g: spans[g] for g in genes})["cross_split_pairs"]
    if residual:
        raise RuntimeError(f"{residual} window pairs overlap across splits")
    stats = {
        "method": "genomic_window_and_protein_cluster_disjoint",
        "n_genes": len(genes),
        "n_window_overlap_edges": len(edges),
        "n_protein_clusters": len({protein_map[g] for g in genes}),
        "n_combined_groups": len(members),
    }
    return {**parts, "family_proportions": family_proportions(df, parts)}, stats
