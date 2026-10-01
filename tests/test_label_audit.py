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
    groups = L.shared_summary_groups(gt)
    sizes = [len(v) for v in groups.values()]
    assert (len(L.templated_genes(gt)), len(groups), sizes[0], len(groups[""])) == (901, 66, 347, 104)
    (largest,) = [ids for ids in groups.values() if len(ids) == 347]
    assert set(gt.set_index("ensembl_id").loc[largest, "symbol"].str[:2]) == {"OR"}
