"""Build the TSS-primary split: disjoint on genomic windows and protein clusters.

The homology split (``data/splits.json``) clusters translated proteins at 40%
identity, but genomically adjacent genes in different protein clusters still
have overlapping 196,608 bp TSS windows. This split groups genes by the union
of window overlap and shared protein cluster (``splits.tss_disjoint``), so no
window can straddle train/val/test.

Inputs are all tracked and stamped into the output with their sha256 (G21):
the window manifest (canonical-TSS spans, ``data/tss_windows.tsv``), the MMseqs2
40% cluster assignments (``data/clusters/homology_id40.tsv``), the family
labels (``data/dataset_dnabert2_meanmean.parquet``), and the gene universe
(``data/splits.json``). The script also writes the cross-split window-overlap
statistics for the homology split and the new split (G2); the new split must
have none, and no protein cluster may straddle it.

Run: uv run scripts/make_tss_disjoint_split.py
Seeds: --seed N --out data/... --leak-out analysis/... (a seed never overwrites
the primary split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cluster.mmseqs_cluster import parse_cluster_tsv
from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, MANIFEST, window_spans
from splits.make_splits import DEFAULT_FRACS, SEED, STRATIFY_COL
from splits.tss_disjoint import build_tss_disjoint
from splits.window_leak import window_leak_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
CLUSTER_TSV = DATA / "clusters" / "homology_id40.tsv"
UNIVERSE = DATA / "splits.json"
FAMILIES = DATA / "dataset_dnabert2_meanmean.parquet"
OUT = DATA / "splits_tss_disjoint.json"
LEAK_OUT = REPO_ROOT / "analysis" / "tss_overlap" / "window_leak.json"


class ResidualOverlap(RuntimeError):
    """A built split still has windows overlapping across splits."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _labels_stamp(path: Path, fams: pd.DataFrame) -> dict:
    """The family labels themselves, hashed: the parquet holding them is rebuilt by
    every extraction, but the labels must not change."""
    pairs = "\n".join(f"{g}\t{f}" for g, f in sorted(zip(fams["ensembl_id"], fams[STRATIFY_COL])))
    return {"path": _stamp(path)["path"], "labels_sha256": hashlib.sha256(pairs.encode()).hexdigest()}


def _stamp(path: Path) -> dict:
    """Path (repo-relative when inside the repo: no local paths in tracked files) and sha256."""
    path = Path(path).resolve()
    shown = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
    return {"path": shown, "sha256": _sha256(path)}


def build(manifest: Path, clusters: Path, families: Path, splits: Path, seed: int,
          out_name: str) -> tuple[dict, dict, dict]:
    """Build the split payload for one seed. Returns (payload, leak stats, parts)."""
    universe = json.loads(splits.read_text())
    genes = sorted(g for s in ("train", "val", "test") for g in universe[s])
    fams = pd.read_parquet(families, columns=["ensembl_id", STRATIFY_COL])
    fams = fams[fams["ensembl_id"].isin(genes)].drop_duplicates("ensembl_id")
    if len(fams) != len(genes):
        raise KeyError(f"{len(genes) - len(fams)} universe genes have no family in {families}")
    spans = window_spans(manifest)
    protein_map = parse_cluster_tsv(clusters)

    parts, stats = build_tss_disjoint(fams, spans, protein_map, seed=seed)
    stats.update(window_length_bp=ENFORMER_WINDOW_LENGTH, protein_min_seq_id=0.40,
                 protein_coverage=0.80)
    payload = {
        "train": parts["train"], "val": parts["val"], "test": parts["test"],
        "seed": seed, "stratify": STRATIFY_COL, "method": "homology_cluster",
        "fracs": list(DEFAULT_FRACS), "cluster_stats": stats,
        "family_proportions": parts["family_proportions"],
        "inputs": {"window_manifest": _stamp(manifest),
                   "cluster_tsv": _stamp(clusters),
                   "gene_universe": _stamp(splits),
                   "families": _labels_stamp(families, fams)},
    }

    leak = {splits.name: window_leak_stats(universe, spans),
            out_name: window_leak_stats(payload, spans)}
    if leak[out_name]["cross_split_pairs"]:
        raise ResidualOverlap(f"{leak[out_name]['cross_split_pairs']} cross-split "
                              "window overlaps remain")
    return payload, leak, parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--clusters", type=Path, default=CLUSTER_TSV)
    ap.add_argument("--families", type=Path, default=FAMILIES,
                    help="parquet with ensembl_id and family")
    ap.add_argument("--splits", type=Path, default=UNIVERSE,
                    help="homology split: gene universe and the leak comparison")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--leak-out", type=Path, default=LEAK_OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.seed != SEED and (args.out.resolve() == OUT.resolve()
                              or args.leak_out.resolve() == LEAK_OUT.resolve()):
        raise ValueError(f"seed {args.seed} is not the primary split: pass --out and --leak-out")

    universe = json.loads(args.splits.read_text())
    payload, leak, parts = build(args.manifest, args.clusters, args.families, args.splits,
                                 args.seed, args.out.name)
    stats = payload["cluster_stats"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    args.leak_out.parent.mkdir(parents=True, exist_ok=True)
    args.leak_out.write_text(json.dumps(
        {**leak, "window_manifest_sha256": payload["inputs"]["window_manifest"]["sha256"]},
        indent=2) + "\n")

    print(f"wrote {args.out.name}: train={len(parts['train'])} val={len(parts['val'])} "
          f"test={len(parts['test'])} ({stats['n_combined_groups']} groups, "
          f"{stats['n_window_overlap_edges']} window-overlap edges)")
    for split, props in parts["family_proportions"].items():
        print(f"    {split:<5s} {props}")
    shared = len(set(parts["test"]) & set(universe["test"]))
    print(f"  test genes shared with {args.splits.name}: {shared}")
    h = leak[args.splits.name]
    print(f"  {args.splits.name}: {h['test_overlapping_trainval']}/{h['n_test']} test genes "
          f"({100 * h['frac_test_overlapping_trainval']:.1f}%) overlap a train/val window; "
          f"{h['cross_split_pairs']} cross-split pairs")
    print(f"  {args.out.name}: 0 cross-split window overlaps -> {args.leak_out}")


if __name__ == "__main__":
    main()
