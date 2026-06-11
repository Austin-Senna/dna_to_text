"""Per-window genomic-feature overlap accounting for TSS windows.

For each TSS-centered window we paint length-``L`` boolean masks of the bases
covered by the target gene's CDS / UTR / exon / gene-span and by neighbour
genes, then report two views:

* **raw fractions** -- fraction of the window overlapping each feature type
  independently (these can overlap, e.g. CDS is a subset of exon);
* a **mutually-exclusive partition** -- every base assigned to exactly one
  bucket by precedence, so the buckets sum to 1.0 per window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, centered_window

UTR_FEATURES = ("five_prime_utr", "three_prime_utr")

# Mutually-exclusive partition buckets, listed in assignment-precedence order.
PARTITION_BUCKETS = (
    "target_cds",
    "target_utr",
    "target_exon_noncoding",
    "target_intron",
    "neighbor_exon",
    "neighbor_intron",
    "intergenic",
)
# Raw (possibly overlapping) per-feature fractions.
RAW_FRACTIONS = (
    "raw_target_cds",
    "raw_target_utr",
    "raw_target_exon",
    "raw_target_intron",
    "raw_neighbor_exon",
    "raw_intergenic",
)


def _paint(mask: np.ndarray, start: int, end: int, w0: int, length: int) -> None:
    """Set ``mask`` True over a 1-based inclusive ``[start, end]`` interval, clipped to the window."""
    a = start - w0
    b = end - w0 + 1
    if a < 0:
        a = 0
    if b > length:
        b = length
    if b > a:
        mask[a:b] = True


def window_overlap(target_id: str, lookup: dict, gtf_index: dict[str, dict]) -> dict:
    """Compute overlap fractions for one gene's TSS window.

    ``lookup`` is the cached Ensembl lookup dict (``seq_region_name``, ``start``,
    ``end``, ``strand``, ...). ``gtf_index`` is the output of
    :func:`tss_overlap.gtf.index_by_chrom`.
    """
    chrom, w0, w1 = centered_window(
        seq_region_name=str(lookup["seq_region_name"]),
        start=int(lookup["start"]),
        end=int(lookup["end"]),
        strand=int(lookup.get("strand", 1)),
    )
    length = w1 - w0 + 1
    result: dict = {
        "ensembl_id": target_id,
        "chrom": chrom,
        "window_start": w0,
        "window_end": w1,
        "window_len": length,
        "biotype": lookup.get("biotype"),
        "in_gtf": False,
        "n_neighbor_genes": 0,
    }

    chrom_idx = gtf_index.get(chrom)
    if chrom_idx is None:
        return result

    # Candidate features: start <= w1 (searchsorted bound) AND end >= w0.
    starts = chrom_idx["start"]
    hi = int(np.searchsorted(starts, w1, side="right"))
    keep = chrom_idx["end"][:hi] >= w0
    idx = np.nonzero(keep)[0]
    feats = chrom_idx["feature"][idx]
    gids = chrom_idx["gene_id"][idx]
    fstart = starts[idx]
    fend = chrom_idx["end"][idx]
    is_target = gids == target_id
    result["in_gtf"] = bool(is_target.any())

    def _mask() -> np.ndarray:
        return np.zeros(length, dtype=bool)

    m_t_cds = _mask()
    m_t_utr = _mask()
    m_t_exon = _mask()
    m_t_gene = _mask()
    m_n_exon = _mask()
    m_n_gene = _mask()
    m_gene_any = _mask()

    for feat, tgt, s, e in zip(feats, is_target, fstart, fend):
        if feat == "gene":
            _paint(m_gene_any, s, e, w0, length)
            _paint(m_t_gene if tgt else m_n_gene, s, e, w0, length)
        elif feat == "exon":
            _paint(m_t_exon if tgt else m_n_exon, s, e, w0, length)
        elif feat == "CDS":
            if tgt:
                _paint(m_t_cds, s, e, w0, length)
        elif feat in UTR_FEATURES:
            if tgt:
                _paint(m_t_utr, s, e, w0, length)

    # Mutually-exclusive partition by precedence.
    assigned = np.zeros(length, dtype=bool)
    parts: dict[str, int] = {}

    def _assign(name: str, mask: np.ndarray) -> None:
        claimed = mask & ~assigned
        parts[name] = int(claimed.sum())
        assigned[claimed] = True

    _assign("target_cds", m_t_cds)
    _assign("target_utr", m_t_utr)
    _assign("target_exon_noncoding", m_t_exon)
    _assign("target_intron", m_t_gene)
    _assign("neighbor_exon", m_n_exon)
    _assign("neighbor_intron", m_n_gene)
    parts["intergenic"] = int(length - int(assigned.sum()))

    for name in PARTITION_BUCKETS:
        result[name] = parts[name] / length

    # Raw (overlapping) fractions.
    raw_intron = m_t_gene & ~m_t_exon
    result["raw_target_cds"] = float(m_t_cds.mean())
    result["raw_target_utr"] = float(m_t_utr.mean())
    result["raw_target_exon"] = float(m_t_exon.mean())
    result["raw_target_intron"] = float(raw_intron.mean())
    result["raw_neighbor_exon"] = float(m_n_exon.mean())
    result["raw_intergenic"] = float((~m_gene_any).mean())

    neighbor_genes = gids[(feats == "gene") & (~is_target)]
    result["n_neighbor_genes"] = int(np.unique(neighbor_genes).size)
    return result


def compute_all(
    meta_df: pd.DataFrame,
    lookup_dir: str | Path,
    gtf_index: dict[str, dict],
    *,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Run :func:`window_overlap` over every gene in ``meta_df``.

    ``meta_df`` must have ``ensembl_id`` (and optionally ``symbol``/``family``).
    Returns ``(per_gene_df, skipped)`` where ``skipped`` lists ``(ensembl_id, reason)``
    for genes without a cached lookup JSON.
    """
    lookup_dir = Path(lookup_dir)
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    records = meta_df.to_dict("records")
    iterator = tqdm(records, desc="tss overlap") if progress else records
    for row in iterator:
        eid = row["ensembl_id"]
        lk_path = lookup_dir / f"{eid}.json"
        if not lk_path.exists():
            skipped.append((eid, "no_lookup"))
            continue
        lookup = json.loads(lk_path.read_text())
        res = window_overlap(eid, lookup, gtf_index)
        res["symbol"] = row.get("symbol")
        res["family"] = row.get("family")
        rows.append(res)

    return pd.DataFrame(rows), skipped
