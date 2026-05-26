"""Build frozen train/val/test splits and write data/splits*.json.

Two split families:
  * ``build_splits``         — the original random, family-stratified 70/15/15.
  * ``build_cluster_splits`` — the homology-aware split required for the
    Bioinformatics submission: whole MMseqs2 protein clusters are assigned to
    a single split (no paralog straddles train/test) while keeping per-family
    proportions close to 70/15/15.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
STRATIFY_COL = "family"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACS = (0.70, 0.15, 0.15)


def build_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    ids = df["ensembl_id"].tolist()
    fams = df[STRATIFY_COL].tolist()

    train_ids, rest_ids, _, rest_fams = train_test_split(
        ids, fams, test_size=0.30, random_state=seed, stratify=fams
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, random_state=seed, stratify=rest_fams
    )

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(ids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_splits_json(
    dataset_path: str | Path,
    out_path: str | Path,
    seed: int = SEED,
) -> dict:
    df = pd.read_parquet(dataset_path)
    parts = build_splits(df, seed=seed)
    payload = {
        **parts,
        "seed": seed,
        "stratify": STRATIFY_COL,
        "source": str(Path(dataset_path).name),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def build_cluster_splits(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    seed: int = SEED,
    fracs: tuple[float, float, float] = DEFAULT_FRACS,
) -> dict[str, list[str]]:
    """Assign whole homology clusters to train/val/test, family-balanced.

    Greedy: process clusters largest-first (they are the binding constraint);
    place each cluster in the split whose per-family quota it least overshoots,
    breaking ties toward the split with the most remaining capacity. No cluster
    is ever split, so paralogous genes stay on one side of the split.
    """
    df = df[["ensembl_id", STRATIFY_COL, cluster_col]]
    families = sorted(df[STRATIFY_COL].unique())
    fam_total = df[STRATIFY_COL].value_counts().to_dict()
    target = {
        s: {fam: fracs[i] * fam_total[fam] for fam in families}
        for i, s in enumerate(SPLIT_NAMES)
    }

    clusters: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for eid, fam, cid in df.itertuples(index=False, name=None):
        clusters[cid].append((eid, fam))

    rng = random.Random(seed)
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)
    cluster_ids.sort(key=lambda c: -len(clusters[c]))  # stable: largest first

    counts = {s: {fam: 0 for fam in families} for s in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for cid in cluster_ids:
        comp = Counter(fam for _, fam in clusters[cid])

        def overshoot(s: str) -> float:
            return sum(
                max(0.0, counts[s][fam] + comp.get(fam, 0) - target[s][fam])
                for fam in families
            )

        def remaining(s: str) -> float:
            return sum(target[s][fam] - counts[s][fam] for fam in families)

        best = min(SPLIT_NAMES, key=lambda s: (overshoot(s), -remaining(s)))
        for eid, _ in clusters[cid]:
            assignment[eid] = best
        for fam in families:
            counts[best][fam] += comp.get(fam, 0)

    parts = {s: [eid for eid, sp in assignment.items() if sp == s] for s in SPLIT_NAMES}

    # Integrity: every gene assigned exactly once, splits disjoint and complete.
    all_ids = set(df["ensembl_id"])
    assert set().union(*[set(parts[s]) for s in SPLIT_NAMES]) == all_ids
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert set(parts[a]).isdisjoint(parts[b]), f"{a}/{b} overlap"
    # No cluster spans splits.
    cid_of = dict(zip(df["ensembl_id"], df[cluster_col]))
    cluster_split: dict[str, str] = {}
    for s in SPLIT_NAMES:
        for eid in parts[s]:
            c = cid_of[eid]
            if c in cluster_split:
                assert cluster_split[c] == s, f"cluster {c} spans splits"
            else:
                cluster_split[c] = s
    return parts


def family_proportions(df: pd.DataFrame, parts: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Per-split family proportions, for verifying balance."""
    fam_of = dict(zip(df["ensembl_id"], df[STRATIFY_COL]))
    out: dict[str, dict[str, float]] = {}
    for s, ids in parts.items():
        fams = [fam_of[e] for e in ids]
        n = len(fams) or 1
        out[s] = {fam: round(cnt / n, 4) for fam, cnt in sorted(Counter(fams).items())}
    return out


def write_cluster_splits_json(
    df: pd.DataFrame,
    out_path: str | Path,
    cluster_stats: dict,
    cluster_col: str = "cluster_id",
    seed: int = SEED,
    fracs: tuple[float, float, float] = DEFAULT_FRACS,
) -> dict:
    """Build a homology-aware split from a df carrying ``cluster_id`` and write JSON."""
    parts = build_cluster_splits(df, cluster_col=cluster_col, seed=seed, fracs=fracs)
    payload = {
        **parts,
        "seed": seed,
        "stratify": STRATIFY_COL,
        "method": "homology_cluster",
        "fracs": list(fracs),
        "cluster_stats": cluster_stats,
        "family_proportions": family_proportions(df, parts),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
