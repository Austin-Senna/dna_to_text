"""The disclosed label and target defects are rules, and the rules give the paper's counts (D5)."""
from __future__ import annotations

import pandas as pd
import pytest

from data_loader import label_audit as L


def _table(rows):
    return pd.DataFrame(rows, columns=["ensembl_id", "symbol", "family", "summary"])


def _hgnc(rows):
    return pd.DataFrame(rows, columns=["ensembl_id", "gene_group"])


def test_noisy_tf_rule():
    gt = _table([("E1", "A", "tf", ""), ("E2", "B", "tf", ""), ("E3", "C", "tf", ""),
                 ("E4", "D", "tf", ""), ("E5", "E", "kinase", "")])
    hg = _hgnc([("E1", "Zinc fingers C2H2-type|KRAB domain containing"),   # a real TF: kept
                ("E2", "Zinc fingers FYVE-type"),                          # ZF only, not C2H2: noisy
                ("E3", "Zinc fingers CCCH-type|Homeoboxes"),               # another TF include: kept
                ("E4", "Ring finger proteins|Zinc fingers RANBP2-type"),   # noisy
                ("E5", "Zinc fingers FYVE-type")])                         # not labelled TF
    assert L.noisy_tf_genes(gt, hg) == {"E2", "E4"}


def test_noisy_tf_rule_refuses_a_tf_without_hgnc_groups():
    gt = _table([("E1", "A", "tf", "")])
    with pytest.raises(ValueError, match="no HGNC row"):
        L.noisy_tf_genes(gt, _hgnc([("E9", "Zinc fingers FYVE-type")]))


def test_noisy_kinase_rule():
    gt = _table([("E1", "A", "kinase", ""), ("E2", "B", "kinase", ""), ("E3", "C", "kinase", ""),
                 ("E4", "D", "kinase", ""), ("E5", "PHKG1", "kinase", ""), ("E6", "F", "kinase", ""),
                 ("E7", "G", "tf", "")])
    hg = _hgnc([("E1", "Receptor tyrosine kinases"),                                # a protein kinase
                ("E2", "A-kinase anchoring proteins"),                              # not a kinase
                ("E3", "Diacylglycerol kinases|C1 domain containing"),             # a lipid kinase
                ("E4", "Protein kinase A subunits|Protein kinase A family"),       # catalytic: kept
                ("E5", "Phosphorylase kinase subunits"),                            # catalytic subunit: kept
                ("E6", "MOB kinase activators|Adenylate kinases"),                 # either: not a kinase wins
                ("E7", "A-kinase anchoring proteins")])                             # not labelled kinase
    assert L.noisy_kinase_genes(gt, hg) == {"not_kinase": {"E2", "E6"}, "small_molecule": {"E3"}}


def test_summary_body_strips_the_symbol_and_provenance():
    a = "Gene Symbol OR1A1 Olfactory receptors interact with odorants. [provided by RefSeq, Jul 2008]"
    b = "Gene Symbol OR2B6 Olfactory receptors interact with odorants. [provided by RefSeq, Jul 2008]"
    assert L.summary_body(a) == L.summary_body(b) == "Olfactory receptors interact with odorants."
    assert L.summary_body(None) == L.summary_body("Gene Symbol X ") == ""


def test_shared_summary_groups():
    gt = _table([("E1", "OR1", "gpcr", "Gene Symbol OR1 Template."),
                 ("E2", "OR2", "gpcr", "Gene Symbol OR2 Template."),
                 ("E3", "T1", "tf", "Gene Symbol T1 "),
                 ("E4", "T2", "tf", None),
                 ("E5", "K1", "kinase", "Gene Symbol K1 Unique text.")])
    assert L.shared_summary_groups(gt) == {"": ["E3", "E4"], "Template.": ["E1", "E2"]}
    assert L.templated_genes(gt) == {"E1", "E2", "E3", "E4"}


@pytest.mark.skipif(not (L.GENE_TABLE.exists() and L.HGNC.exists()),
                    reason="Stage 1 inputs are not tracked")
def test_the_real_counts():
    gt, hg, _ = L.load_inputs()
    assert len(L.noisy_tf_genes(gt, hg)) == 387
    kin = L.noisy_kinase_genes(gt, hg)
    assert (len(kin["not_kinase"]), len(kin["small_molecule"])) == (92, 58)
    # Every listed group still exists under that name among the kinase-labelled genes.
    labelled = hg[hg["ensembl_id"].isin(gt.loc[gt["family"] == "kinase", "ensembl_id"])]
    seen = {x for text in labelled["gene_group"].fillna("") for x in text.split("|")}
    assert (L.NOT_KINASE_GROUPS | L.SMALL_MOLECULE_KINASE_GROUPS) <= seen
    assert {"PHKG1", "PHKG2"} <= set(gt.loc[gt["family"] == "kinase", "symbol"])
    groups = L.shared_summary_groups(gt)
    sizes = [len(v) for v in groups.values()]
    assert (len(L.templated_genes(gt)), len(groups), sizes[0], len(groups[""])) == (901, 66, 347, 104)
    (largest,) = [ids for ids in groups.values() if len(ids) == 347]
    assert set(gt.set_index("ensembl_id").loc[largest, "symbol"].str[:2]) == {"OR"}
