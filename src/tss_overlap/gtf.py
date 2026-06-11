"""Parse an Ensembl GTF into a compact feature table and per-chromosome index.

Coordinates are kept in GTF convention: 1-based, inclusive on both ends. This
matches ``data_loader.enformer_windows.centered_window`` so window/feature
intersections need no offset juggling.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Feature rows we keep for window-overlap accounting.
KEEP_FEATURES = ("gene", "exon", "CDS", "five_prime_utr", "three_prime_utr")
# Standard nuclear + mitochondrial chromosomes (Ensembl naming, no "chr" prefix).
STANDARD_CHROMS = tuple(str(c) for c in range(1, 23)) + ("X", "Y", "MT")

_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')


def load_gtf_features(
    gtf_path: str | Path,
    cache_path: str | Path | None = None,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load GTF rows for :data:`KEEP_FEATURES` as ``[chrom, feature, gene_id, start, end]``.

    ``start``/``end`` are 1-based inclusive. The parsed frame is cached to
    ``cache_path`` (parquet) and reused unless ``rebuild`` is set, so the ~52 MB
    GTF is only parsed once.
    """
    gtf_path = Path(gtf_path)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    raw = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 2, 3, 4, 8],
        names=["chrom", "feature", "start", "end", "attrs"],
        dtype={"chrom": str, "feature": str, "start": np.int64, "end": np.int64, "attrs": str},
        compression="infer",
    )
    raw = raw[raw["feature"].isin(KEEP_FEATURES) & raw["chrom"].isin(STANDARD_CHROMS)]
    raw = raw.assign(gene_id=raw["attrs"].str.extract(_GENE_ID_RE, expand=False))
    df = (
        raw[["chrom", "feature", "gene_id", "start", "end"]]
        .dropna(subset=["gene_id"])
        .reset_index(drop=True)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def index_by_chrom(df: pd.DataFrame) -> dict[str, dict]:
    """Per-chromosome sorted numpy arrays for fast windowed slicing.

    Rows are sorted by ``start`` so a window query can bound candidates with
    ``np.searchsorted`` on the ``start`` array, then filter on ``end``.
    """
    index: dict[str, dict] = {}
    for chrom, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start", kind="stable")
        index[str(chrom)] = {
            "start": sub["start"].to_numpy(np.int64),
            "end": sub["end"].to_numpy(np.int64),
            "feature": sub["feature"].to_numpy(object),
            "gene_id": sub["gene_id"].to_numpy(object),
        }
    return index
