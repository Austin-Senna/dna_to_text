"""Table rules that a wrong split or a wrong comparison would silently change (G25, G26)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_paper_tables as bt  # noqa: E402


def test_the_split_comparison_reads_each_arms_primary_split(monkeypatch):
    """G25: TSS rows compare random against the disjoint split, not the homology
    split, whose TSS windows overlap across partitions."""
    value = {"hom": 0.111, "dis": 0.222, "rnd": 0.0}
    monkeypatch.setattr(bt, "HOM", "hom")
    monkeypatch.setattr(bt, "DIS", "dis")
    monkeypatch.setattr(bt, "RND", "rnd")
    monkeypatch.setattr(bt, "_value", lambda split, arm, task, name: value[split])
    monkeypatch.setattr(bt, "_nt_label", lambda: "nt")
    cds, tss = bt.build_split_comparison().split("TSS window")
    assert "0.111" in cds and "0.222" not in cds
    assert "0.222" in tss and "0.111" not in tss


class _Split:
    def __init__(self, by):
        self.by = by

    def cells(self, arm, task):
        return self.by

    def cell(self, arm, task, src):
        return self.by[src]

    def best(self, enc, arm, task):
        return self.by[f"tss_{enc}_meanmean"]

    def anchored_composition(self, enc, task):
        return f"tss_{enc}_chunk4mergc"


def _anchored_fixture(monkeypatch, lo, delta=0.1):
    """Every anchored cell 0.40 against a whole window of 0.30; each paired test
    has interval [lo, 0.2] and point ``delta``."""
    per, expl = {}, {}
    for split in (bt.CDS, bt.TSS):
        by = per[split] = {}
        for enc in bt.ENCODERS:
            for src, v in ((f"tss_{enc}_meanmean", 0.30), (f"tss_{enc}_tssanchored", 0.40),
                           (f"tss_{enc}_chunk4mergc", 0.2), (f"tss_{enc}_chunk6mer", 0.1)):
                by[src] = {"key": f"{split}/{src}", bt.F1: v, "C_sweep": [{"macro_f1": v, "C": 1.0}]}
            expl[f"{split} family5: {enc} anchored > whole-window"] = {
                "a": f"{split}/tss_{enc}_tssanchored", "b": f"{split}/tss_{enc}_meanmean",
                "delta_point": delta, "delta_ci95": [lo, 0.2]}
        for src, v in ((bt.ENF_WHOLE, 0.3), (bt.ENF_CENTRE, 0.4)):
            by[src] = {"key": f"{split}/{src}", bt.F1: v}
        expl[f"{split} family5: Enformer centre > whole"] = {
            "a": f"{split}/{bt.ENF_CENTRE}", "b": f"{split}/{bt.ENF_WHOLE}",
            "delta_point": delta, "delta_ci95": [lo, 0.2]}
    monkeypatch.setattr(bt, "HOM", _Split(per[bt.CDS]))
    monkeypatch.setattr(bt, "DIS", _Split(per[bt.TSS]))
    monkeypatch.setattr(bt, "STATS", {"exploratory": expl, "intervals": {
        f"{s}/family5/{src}": {"point": 0.4, "ci95": [0.35, 0.45]}
        for s in (bt.CDS, bt.TSS) for src in [*(f"tss_{e}_tssanchored" for e in bt.ENCODERS), bt.ENF_CENTRE]}})


@pytest.mark.parametrize("lo,bolded", [(0.001, True), (-0.001, False)])
def test_anchored_bolding_follows_the_paired_test(monkeypatch, lo, bolded):
    """G26: bold only when the paired anchored - whole interval excludes 0, even
    if the anchored CI alone clears the whole-window point."""
    _anchored_fixture(monkeypatch, lo)
    # Each anchored CI alone clears its whole-window point: the old rule always bolded.
    assert ("\\textbf{0.400" in bt.build_tss_anchored()) is bolded


def test_anchored_delta_column_is_the_paired_test(monkeypatch):
    _anchored_fixture(monkeypatch, -0.05)
    rows = [r for r in bt.build_tss_anchored().splitlines() if r.startswith(("DNABERT-2", "Enformer"))]
    assert rows and all("& +0.100 [-0.050, +0.200] &" in r for r in rows)


def test_anchored_delta_must_be_the_table_difference(monkeypatch):
    """G8: a paired point off the displayed cells' difference is another fit."""
    _anchored_fixture(monkeypatch, 0.001, delta=0.08)
    with pytest.raises(bt.R.MixedRecords, match="not the table's difference"):
        bt.build_tss_anchored()
