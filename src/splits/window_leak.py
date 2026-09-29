"""Cross-split TSS-window overlap: the genomic half of G2.

Two genes whose windows overlap on the chromosome share input bases, so a pair
in different splits leaks the TSS arm. ``window_leak_stats`` counts those pairs
and the test genes they touch; it is the tracked computation behind the
appendix's cross-split figure (235 of 487 test genes, 48.3%, on the May windows
and the homology split).
"""
from __future__ import annotations

from collections import defaultdict

Span = tuple[str, int, int]  # (chrom, start, end), 1-based inclusive
SPLITS = ("train", "val", "test")


def overlap_pairs(spans: dict[str, Span]) -> list[tuple[str, str, int]]:
    """Every pair of genes whose spans overlap, with the overlap in bases."""
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for gene, (chrom, start, end) in spans.items():
        by_chrom[chrom].append((start, end, gene))
    pairs: list[tuple[str, str, int]] = []
    for recs in by_chrom.values():
        recs.sort()
        for i, (s_i, e_i, g_i) in enumerate(recs):
            for s_j, e_j, g_j in recs[i + 1:]:
                if s_j > e_i:  # sorted by start: nothing further overlaps i
                    break
                pairs.append((g_i, g_j, min(e_i, e_j) - s_j + 1))
    return pairs


def window_leak_stats(split: dict, spans: dict[str, Span]) -> dict:
    """Cross-split window overlap for one split file over the genes it assigns."""
    split_of = {g: s for s in SPLITS for g in split[s]}
    missing = sorted(set(split_of) - set(spans))
    if missing:
        raise KeyError(f"{len(missing)} split genes have no window span: {missing[:5]}")
    kinds: dict[str, int] = defaultdict(int)
    touching: dict[str, set[str]] = {"train": set(), "val": set()}
    leakers: set[str] = set()
    for a, b, _ in overlap_pairs({g: spans[g] for g in split_of}):
        sa, sb = split_of[a], split_of[b]
        if sa == sb:
            continue
        kinds["_".join(sorted((sa, sb), key=SPLITS.index)[::-1])] += 1
        for x, sx, sy in ((a, sa, sb), (b, sb, sa)):
            if sx == "test":
                touching[sy].add(x)
            elif sy == "test":
                leakers.add(x)
    n_test = len(split["test"])
    either = touching["train"] | touching["val"]
    return {
        "cross_split_pairs": sum(kinds.values()),
        "pairs_test_train": kinds["test_train"],
        "pairs_train_val": kinds["val_train"],
        "pairs_test_val": kinds["test_val"],
        "n_test": n_test,
        "test_overlapping_trainval": len(either),
        "test_overlapping_train": len(touching["train"]),
        "test_overlapping_val": len(touching["val"]),
        "frac_test_overlapping_trainval": len(either) / n_test if n_test else 0.0,
        "trainval_genes_overlapping_test": len(leakers),
    }
