"""CLI: genomic-feature overlap of TSS windows (#4a).

For every TSS gene, compute how much of its 196,608 bp encoder window overlaps
the target gene's CDS / UTR / exon / intron, neighbouring exons/introns, and
intergenic space. Emits a per-gene table, a per-family summary (mutually-
exclusive partition + raw overlapping fractions), and two figures.

Gene set + family labels come from a TSS encoder parquet (meta is identical
across encoders). Genomic coordinates come from the cached Ensembl lookups in
``data/enformer_windows/_lookup`` and the Ensembl GTF in ``data/annotation``.
All inputs are local -- no network.

Run: ``uv run scripts/tss_overlap.py``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tss_overlap import compute_all, index_by_chrom, load_gtf_features  # noqa: E402
from tss_overlap.overlap import PARTITION_BUCKETS, RAW_FRACTIONS  # noqa: E402

DATA = REPO_ROOT / "data"

# Human-readable labels for the partition stack.
PARTITION_LABELS = {
    "target_cds": "target CDS",
    "target_utr": "target UTR",
    "target_exon_noncoding": "target exon (non-coding)",
    "target_intron": "target intron",
    "neighbor_exon": "neighbour exon",
    "neighbor_intron": "neighbour intron",
    "intergenic": "intergenic",
}


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in (row[c] for c in cols)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_table(df: pd.DataFrame, name: str, tables_dir: Path, *, title: str, description: str,
                 markdown: bool = True) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    if markdown:
        (tables_dir / f"{name}.md").write_text(f"# {title}\n\n{description}\n\n{_to_markdown(df)}")


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarise(per_gene: pd.DataFrame) -> pd.DataFrame:
    """Mean partition + raw fractions, one row per family plus an 'overall' row (in_gtf genes only)."""
    df = per_gene[per_gene["in_gtf"]].copy()
    metric_cols = list(PARTITION_BUCKETS) + list(RAW_FRACTIONS) + ["n_neighbor_genes"]
    rows = []
    for family in sorted(df["family"].dropna().unique()):
        sub = df[df["family"] == family]
        rows.append({"group": family, "n_genes": len(sub), **{c: float(sub[c].mean()) for c in metric_cols}})
    rows.insert(0, {"group": "overall", "n_genes": len(df),
                    **{c: float(df[c].mean()) for c in metric_cols}})
    return pd.DataFrame(rows)


def plot_partition_by_family(summary: pd.DataFrame, path: Path) -> None:
    groups = summary["group"].tolist()
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(groups) + 1.5))
    cmap = plt.get_cmap("tab10")
    left = np.zeros(len(groups))
    for i, bucket in enumerate(PARTITION_BUCKETS):
        vals = summary[bucket].to_numpy()
        ax.barh(y, vals, left=left, color=cmap(i % 10), label=PARTITION_LABELS[bucket])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean fraction of 196,608 bp window")
    ax.set_title("TSS-window composition by gene family (mutually-exclusive partition)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    _savefig(fig, path)


def plot_intergenic_intron_distribution(per_gene: pd.DataFrame, path: Path) -> None:
    df = per_gene[per_gene["in_gtf"]]
    families = sorted(df["family"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(families) + 2), sharey=True)
    for ax, col, title in (
        (axes[0], "intergenic", "intergenic fraction"),
        (axes[1], "target_intron", "target-intron fraction"),
    ):
        ax.boxplot([df[df["family"] == fam][col].to_numpy() for fam in families],
                   vert=False, showfliers=False)
        ax.set_yticks(np.arange(1, len(families) + 1))
        ax.set_yticklabels(families)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
    axes[0].set_title("Per-window fraction distributions by family")
    _savefig(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset_tss_dnabert2.parquet",
                    help="TSS encoder parquet for the gene/family table")
    ap.add_argument("--lookup-dir", type=Path, default=DATA / "enformer_windows" / "_lookup")
    ap.add_argument("--gtf", type=Path, default=DATA / "annotation" / "Homo_sapiens.GRCh38.110.gtf.gz")
    ap.add_argument("--gtf-cache", type=Path, default=DATA / "annotation" / "gtf_features.parquet")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "analysis" / "tss_overlap")
    ap.add_argument("--rebuild-gtf", action="store_true", help="re-parse the GTF even if a cache exists")
    args = ap.parse_args()

    meta = pd.read_parquet(args.dataset, columns=["ensembl_id", "symbol", "family"])
    print(f"genes in dataset: {len(meta)}")

    print("loading GTF features ...")
    gtf = load_gtf_features(args.gtf, args.gtf_cache, rebuild=args.rebuild_gtf)
    print(f"  {len(gtf):,} feature rows across {gtf['chrom'].nunique()} chromosomes")
    gtf_index = index_by_chrom(gtf)

    per_gene, skipped = compute_all(meta, args.lookup_dir, gtf_index)

    # Partition buckets are leftover-complete, so they must sum to 1.0 per window.
    part_sums = per_gene[list(PARTITION_BUCKETS)].sum(axis=1)
    if not np.allclose(part_sums.to_numpy(), 1.0, atol=1e-6):
        bad = int((~np.isclose(part_sums.to_numpy(), 1.0, atol=1e-6)).sum())
        raise SystemExit(f"partition buckets do not sum to 1.0 for {bad} windows")

    n_in_gtf = int(per_gene["in_gtf"].sum())
    tables_dir = args.out / "tables"
    figures_dir = args.out / "figures"

    _write_table(
        per_gene, "per_gene_overlap", tables_dir,
        title="TSS-window genomic overlap (per gene)",
        description="Per-gene overlap of the 196,608 bp TSS window: mutually-exclusive "
                    "partition buckets and raw (overlapping) per-feature fractions.",
        markdown=False,
    )
    summary = summarise(per_gene)
    _write_table(
        summary, "overlap_by_family", tables_dir,
        title="TSS-window composition by gene family",
        description="Mean per-window partition (sums to 1.0) and raw overlapping fractions, "
                    f"averaged over {n_in_gtf} genes found in the GTF.",
    )

    plot_partition_by_family(summary, figures_dir / "partition_by_family.png")
    plot_intergenic_intron_distribution(per_gene, figures_dir / "intergenic_intron_distribution.png")

    coding = per_gene.loc[per_gene["in_gtf"], "target_cds"].mean()
    intergenic = per_gene.loc[per_gene["in_gtf"], "intergenic"].mean()
    intron = per_gene.loc[per_gene["in_gtf"], "target_intron"].mean()
    print(f"\nprocessed {len(per_gene)} genes ({n_in_gtf} found in GTF); skipped {len(skipped)}")
    if skipped:
        print(f"  skipped (no cached lookup): {[e for e, _ in skipped][:10]}{' ...' if len(skipped) > 10 else ''}")
    print(f"mean window composition (in-GTF genes): "
          f"target-CDS={coding:.4f}, target-intron={intron:.4f}, intergenic={intergenic:.4f}")
    print(f"wrote tables -> {tables_dir}")
    print(f"wrote figures -> {figures_dir}")


if __name__ == "__main__":
    main()
