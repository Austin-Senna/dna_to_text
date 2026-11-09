"""Known label and target defects, as gene sets the statistics can mask (D5).

Three defects the paper discloses, each defined here by a rule over HGNC groups:

* **Noisy TF labels.** The TF family includes any HGNC group whose name contains
  "zinc finger" (``dataset_loader.FAMILIES``). C2H2 zinc fingers are the classic
  DNA-binding TFs; the other zinc-finger groups (CCCH, FYVE, RING, DHHC, PHD, ...)
  mostly are not. A TF-labelled gene is noisy when "zinc finger" is its only TF
  include match and none of its groups is C2H2-type.
* **Noisy kinase labels.** The kinase family includes any HGNC group whose name
  matches ``\bkinase``. Some such groups hold no protein kinase: anchoring and
  scaffold proteins (AKAPs, membrane-associated guanylate kinases), phosphatases,
  activators, regulatory or non-catalytic subunits and complex partners
  (``NOT_KINASE_GROUPS``), and kinases of small molecules (lipids, nucleotides,
  sugars, metabolites; ``SMALL_MOLECULE_KINASE_GROUPS``). A kinase-labelled gene is
  noisy when every kinase group it belongs to is one of these, except the
  catalytic subunits that HGNC files only under a subunit group
  (``CATALYTIC_SUBUNITS``). Found by the Rule 3 audit (Oct 1).
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
KINASE = re.compile(r"\bkinase", re.IGNORECASE)
NOT_KINASE_GROUPS = frozenset({
    "A-kinase anchoring proteins", "MAP kinase phosphatases", "Membrane associated guanylate kinases",
    "MOB kinase activators", "Phosphorylase kinase subunits", "Protein kinase A subunits",
    "Protein kinase AMP-activated non-catalytic subunit gamma family",
    "Protein kinase AMP-activated non-catalytic subunit beta family", "CDK activating kinase complex",
    "PI4KA lipid kinase complex"})
SMALL_MOLECULE_KINASE_GROUPS = frozenset({
    "Diacylglycerol kinases", "Adenylate kinases", "Phosphatidylinositol 3-kinase family",
    "Pantothenate kinase family", "6-phosphofructo-2-kinase/fructose-2,6-biphosphatase family",
    "Deoxyribonucleoside kinases", "Glycerol kinase family", "Uridine-cytidine kinase family",
    "Ethanolamine kinase family", "Pyruvate kinase family", "Sphingosine kinase family",
    "Choline kinase family", "Creatine kinase family"})
# Phosphorylase kinase's gamma subunits are its catalytic protein kinases.
CATALYTIC_SUBUNITS = frozenset({"PHKG1", "PHKG2"})
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


def _kinase_includes() -> list[str]:
    (includes,) = [inc for name, _, inc, _ in FAMILIES if name == "kinase"]
    if includes != [KINASE.pattern]:
        raise RuntimeError(f"the kinase family's includes changed to {includes}; the noisy-label rule is stale")
    return includes


def noisy_kinase_genes(gene_table: pd.DataFrame, hgnc: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Kinase-labelled genes that are not protein kinases, by tier: ``not_kinase``
    (no kinase activity of their own) and ``small_molecule`` (kinases, but not of
    proteins). A gene is in the first tier if any of its kinase groups is."""
    _kinase_includes()
    kin = gene_table[gene_table["family"] == "kinase"]
    groups = hgnc[hgnc["ensembl_id"].isin(kin["ensembl_id"])]
    if groups["ensembl_id"].duplicated().any():
        raise ValueError("an Ensembl id maps to more than one HGNC row")
    missing = set(kin["ensembl_id"]) - set(groups["ensembl_id"])
    if missing:
        raise ValueError(f"{len(missing)} kinase genes have no HGNC row, e.g. {sorted(missing)[:3]}")
    known = NOT_KINASE_GROUPS | SMALL_MOLECULE_KINASE_GROUPS
    symbol = dict(zip(gene_table["ensembl_id"], gene_table["symbol"]))
    tiers: dict[str, set[str]] = {"not_kinase": set(), "small_molecule": set()}
    for eid, text in zip(groups["ensembl_id"], groups["gene_group"].fillna("")):
        hits = [x for x in text.split("|") if KINASE.search(x)]
        if not hits or not set(hits) <= known or symbol[eid] in CATALYTIC_SUBUNITS:
            continue
        tiers["not_kinase" if set(hits) & NOT_KINASE_GROUPS else "small_molecule"].add(eid)
    return {k: frozenset(v) for k, v in tiers.items()}


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
