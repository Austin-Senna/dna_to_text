"""Known label and target defects, as gene sets the statistics can mask (D5).

Two defects the paper discloses, each defined here by a rule rather than a list:

* **Noisy TF labels.** The TF family includes any HGNC group whose name contains
  "zinc finger" (``dataset_loader.FAMILIES``). C2H2 zinc fingers are the classic
  DNA-binding TFs; the other zinc-finger groups (CCCH, FYVE, RING, DHHC, PHD, ...)
  mostly are not. A TF-labelled gene is noisy when "zinc finger" is its only TF
  include match and none of its groups is C2H2-type.
* **Templated GenePT targets.** GenePT embeds each gene's summary text. Genes whose
  summary body (without the leading ``Gene Symbol <sym>`` and the trailing
  ``[provided by ...]`` tag) is shared with another gene get targets that differ
  only by the symbol: the olfactory receptors' one template, a GO "Predicted to
  enable ..." template, and the genes with no summary at all.

Inputs: ``data/gene_table.parquet`` (labels, summaries) and
``data/hgnc/hgnc_complete_set.tsv`` (group names), both built by Stage 1.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from data_loader.dataset_loader import FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[2]
GENE_TABLE = REPO_ROOT / "data" / "gene_table.parquet"
HGNC = REPO_ROOT / "data" / "hgnc" / "hgnc_complete_set.tsv"

ZINC_FINGER = r"zinc finger"
C2H2 = re.compile(r"C2H2", re.IGNORECASE)
_PREFIX = re.compile(r"^Gene Symbol \S+\s*")
_PROVENANCE = re.compile(r"\[provided by [^\]]*\]")


def _tf_includes() -> list[str]:
    (includes,) = [inc for name, _, inc, _ in FAMILIES if name == "tf"]
    if ZINC_FINGER not in includes:
        raise RuntimeError("the TF family no longer includes 'zinc finger'; the noisy-label rule is stale")
    return includes


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(gene_table: Path = GENE_TABLE, hgnc: Path = HGNC) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The gene table, the HGNC groups (ensembl_id, gene_group) and the inputs' sha256s."""
    gt = pd.read_parquet(gene_table, columns=["ensembl_id", "symbol", "family", "summary"])
    hg = pd.read_csv(hgnc, sep="\t", usecols=["ensembl_gene_id", "gene_group"], dtype=str)
    hg = hg.rename(columns={"ensembl_gene_id": "ensembl_id"}).dropna(subset=["ensembl_id"])
    stamp = {"gene_table": _sha(gene_table), "hgnc": _sha(hgnc)}
    return gt, hg, stamp


def noisy_tf_genes(gene_table: pd.DataFrame, hgnc: pd.DataFrame) -> frozenset[str]:
    """TF-labelled genes that are TFs only through a non-C2H2 zinc-finger group."""
    tf = gene_table[gene_table["family"] == "tf"]
    groups = hgnc[hgnc["ensembl_id"].isin(tf["ensembl_id"])]
    if groups["ensembl_id"].duplicated().any():
        raise ValueError("an Ensembl id maps to more than one HGNC row")
    missing = set(tf["ensembl_id"]) - set(groups["ensembl_id"])
    if missing:
        raise ValueError(f"{len(missing)} TF genes have no HGNC row, e.g. {sorted(missing)[:3]}")
    zf = re.compile(ZINC_FINGER, re.IGNORECASE)
    other = re.compile("|".join(p for p in _tf_includes() if p != ZINC_FINGER), re.IGNORECASE)
    noisy = set()
    for eid, text in zip(groups["ensembl_id"], groups["gene_group"].fillna("")):
        if zf.search(text) and not other.search(text) and not C2H2.search(text):
            noisy.add(eid)
    return frozenset(noisy)


def summary_body(summary: str | None) -> str:
    """The summary without its per-gene symbol prefix and provenance tag."""
    text = summary if isinstance(summary, str) else ""   # None or NaN: no summary
    return _PROVENANCE.sub("", _PREFIX.sub("", text)).strip()


def shared_summary_groups(gene_table: pd.DataFrame) -> dict[str, list[str]]:
    """Summary body -> the genes that share it, for every body shared by two or more
    genes (the empty body included), largest group first."""
    by_body: dict[str, list[str]] = {}
    for eid, s in zip(gene_table["ensembl_id"], gene_table["summary"]):
        by_body.setdefault(summary_body(s), []).append(eid)
    shared = {b: sorted(ids) for b, ids in by_body.items() if len(ids) > 1}
    return dict(sorted(shared.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def templated_genes(gene_table: pd.DataFrame) -> frozenset[str]:
    """Every gene whose GenePT target comes from a summary shared with another gene."""
    return frozenset(g for ids in shared_summary_groups(gene_table).values() for g in ids)
