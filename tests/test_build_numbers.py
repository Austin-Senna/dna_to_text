"""numbers.tex (Phase 6): every number the prose states, under a key named for what
it measures, so no prose decimal is hand-transcribed. The prose reads a value as
``\\val{key}``; an undefined key stops the LaTeX build."""
from __future__ import annotations

import json
import re

import pytest

import build_numbers as bn
import build_paper_tables as bt
from linear_trainer import records as R


@pytest.fixture(scope="module")
def numbers():
    return bn.build()


def test_keys_are_well_formed_and_values_are_latex_safe(numbers):
    assert numbers and all(bn.KEY.fullmatch(k) for k in numbers), [k for k in numbers if not bn.KEY.fullmatch(k)]
    bad = {k: v for k, v in numbers.items() if re.search(r"(?<![\w{])-\d|[%#&_^~]", v)}
    assert not bad, bad                            # a bare hyphen-minus or an unescaped special character


def test_a_value_is_its_source_formatted(numbers):
    """Spot checks read straight from the inputs, not through the builder."""
    s = json.loads((R.V2 / "statistics.json").read_text())
    t1 = s["confirmatory"]["T1 encoder > nucleotide k-mer"]
    assert numbers["t1.delta"] == bn.latex(bt.sgn(t1["delta_point"], 3))
    assert numbers["t1.ci"] == bn.latex(f"{bt.sgn(t1['delta_ci95'][0], 3)}, {bt.sgn(t1['delta_ci95'][1], 3)}")
    assert numbers["t1.p-holm"] == f"{t1['p_holm']:.3f}"
    # `.p` is the unadjusted one-sided p in every family; Holm is always spelled out (review, Oct 2).
    assert numbers["t1.p"] == f"{t1['p_one_sided']:.3f}"
    mt1 = s["sensitivity"]["label_noise"]["tests"]["T1 encoder > nucleotide k-mer"]
    assert numbers["masked.label-noise.t1.p"] == f"{mt1['p_one_sided']:.3f}" and "masked.label-noise.t1.p-holm" not in numbers
    t4 = s["sensitivity"]["label_noise"]["tests"]["T4 CDS > TSS (same encoder)"]
    assert numbers["masked.label-noise.excluded-dis"] == str(t4["n_excluded"])
    c = json.loads((R.V2 / "counts.json").read_text())
    assert numbers["n.noisy-tf"] == str(c["noisy_tf_labels"]["n"]["all genes"])
    assert numbers["n.template.largest"] == str(c["templated_summaries"]["largest_group"]["n"])
    hom = {r["feature_source"]: r for r in R.load(bt.CDS).values() if r["task"] == "family5" and r["arm"] == "cds"}
    assert numbers["cell.hom.cds.f5.aa3"] == bt.f(hom["aa3"][bt.F1], 3)
    leak = json.loads((R.REPO_ROOT / "analysis" / "tss_overlap" / "window_leak.json").read_text())["splits.json"]
    assert numbers["tss-overlap.test"] == str(leak["test_overlapping_trainval"])
    assert numbers["purge.cds.test"] == str(len(next(iter(hom.values()))["purge"]["test_masked"]))
    ex = s["exploratory"]["splits.json genept: esm2_650m > aa_kmer"]
    assert numbers["ex.hom.gp.esm2-650m-gt-aa-kmer.delta"] == bn.latex(bt.sgn(ex["delta_point"], 3))
    assert numbers["ex.hom.gp.esm2-650m-gt-aa-kmer.p"] == f"{ex['p_one_sided']:.3f}"


def test_the_picks_are_the_validation_picks(numbers):
    """Independent of the builder's pick path: the argmax of the validation score
    over every encoder cell. On CDS family5 it differs from the test argmax, so a
    pick made on test scores fails here."""
    from linear_trainer.selection import val_score
    for split, arm, key in ((bt.CDS, "cds", "cds.f5"), (bt.TSS, "tss", "tss.f5")):
        cells = [r for r in R.load(split).values() if (r["arm"], r["task"]) == (arm, "family5")
                 and not r["shuffled_labels"] and any(r["feature_source"].removeprefix("tss_").startswith(e + "_")
                                                      for e in bt.ENCODERS)
                 and r["feature_source"].rsplit("_", 1)[1] in {p for e in bt.ENCODERS
                                                               for p in bt.encoder_pools(e, arm.upper())}]
        by_val = max(cells, key=val_score)
        assert numbers[f"{key}.best-encoder"] == bt.ENC_DISPLAY[R.encoder_of(by_val["feature_source"])]
        if arm == "cds":
            assert max(cells, key=lambda r: r[bt.F1])["key"] != by_val["key"]   # the case that discriminates


def test_a_value_that_moves_under_a_perturbation_is_marked(numbers):
    """The TSS null median moves 0.1675 -> 0.1678 on the kernel side (Oct 2)."""
    assert numbers["null.dis.f5.enformer-tss-4mer.median"] == f"0.167{bt.MARK}"
    assert bt.MARK not in numbers["t1.delta"]       # no confirmatory side moves


def test_every_val_in_the_paper_is_defined():
    keys = set(bn.build())
    assert bn.undefined(bn.paper_sources(), keys) == []


def test_an_undefined_or_pending_site_is_reported():
    texts = {"a.tex": r"x \val{t1.delta} y \val{t9.delta} z \pending{0.224}"}
    assert bn.undefined(texts, {"t1.delta"}) == [("a.tex", "t9.delta")]
    assert bn.pending(texts) == [("a.tex", "0.224")]


def test_the_window_overlap_is_computed_on_the_windows_the_records_used(monkeypatch, tmp_path):
    """Computed at build time from the current split and manifest (no stale file);
    a manifest other than the one the records' window purge read is refused."""
    p = tmp_path / "tss_windows.tsv"
    lines = bn.MANIFEST.read_text().splitlines(keepends=True)
    p.write_text("".join(lines[:-1]))
    monkeypatch.setattr(bn, "MANIFEST", p)
    with pytest.raises(R.MixedRecords, match="window manifest"):
        bn.build()


def test_the_window_composition_is_the_audit_tables_overall_row(numbers):
    """Appendix overlap audit and Results' "~1% coding": the mean target-CDS share
    over every window, read from scripts/tss_overlap.py's table."""
    import csv
    with (bn.AUDIT / "overlap_by_family.csv").open() as fh:
        overall = next(r for r in csv.DictReader(fh) if r["group"] == "overall")
    assert numbers["tss-comp.target-cds"] == f"{float(overall['target_cds']):.3f}"
    assert numbers["tss-comp.target-cds-pct"] == f"{float(overall['target_cds']) * 100:.0f}"
    assert numbers["tss-comp.neighbor-exon-pct"] == f"{float(overall['neighbor_exon']) * 100:.0f}"
    buckets = json.loads((bn.AUDIT / "provenance.json").read_text())["partition"]
    assert sum(float(overall[b]) for b in buckets) == pytest.approx(1.0)   # the prose's parts cover the window
    assert numbers["tss-comp.n"] == bn.count(int(overall["n_genes"]))


@pytest.mark.parametrize("field", ["manifest_sha256", "gtf_sha256"])
def test_a_composition_audit_older_than_the_windows_is_refused(monkeypatch, tmp_path, field):
    prov = json.loads((bn.AUDIT / "provenance.json").read_text())
    (tmp_path / "provenance.json").write_text(json.dumps({**prov, field: "0" * 64}))
    (tmp_path / "overlap_by_family.csv").write_text((bn.AUDIT / "overlap_by_family.csv").read_text())
    monkeypatch.setattr(bn, "AUDIT", tmp_path)
    with pytest.raises(R.MixedRecords, match="composition audit"):
        bn.build()


def test_a_composition_audit_on_another_gene_set_is_refused(monkeypatch, tmp_path):
    (tmp_path / "provenance.json").write_text((bn.AUDIT / "provenance.json").read_text())
    table = (bn.AUDIT / "overlap_by_family.csv").read_text().splitlines(keepends=True)
    head, overall = table[0], table[1].split(",")
    assert overall[0] == "overall"
    overall[1] = str(int(overall[1]) - 1)
    (tmp_path / "overlap_by_family.csv").write_text(head + ",".join(overall) + "".join(table[2:]))
    monkeypatch.setattr(bn, "AUDIT", tmp_path)
    with pytest.raises(R.MixedRecords, match="genes"):
        bn.build()


def test_the_scored_overlap_counts_only_the_genes_left_after_the_purge(numbers):
    """Table A13 states the overlap on the test genes its homology-split F1 scores
    (Hayden, Oct 2): the partition minus the evaluation purge."""
    from data_loader.enformer_windows import window_spans
    from splits.window_leak import window_leak_stats
    split = json.loads((R.REPO_ROOT / "data" / bt.CDS).read_text())
    rec = next(r for r in R.load(bt.CDS).values() if r["arm"] == "tss")
    scored = {**split, "test": [g for g in split["test"] if g not in set(rec["purge"]["test_masked"])]}
    want = window_leak_stats(scored, window_spans(bn.MANIFEST))
    assert numbers["tss-overlap.scored.n-test"] == str(len(split["test"]) - len(rec["purge"]["test_masked"]))
    assert numbers["tss-overlap.scored.test"] == str(want["test_overlapping_trainval"])
    assert int(numbers["tss-overlap.scored.test"]) < int(numbers["tss-overlap.test"])


def test_the_tex_file_defines_every_key(numbers):
    tex = bn.render(numbers)
    assert sorted(re.findall(r"\\@namedef\{mina@([^}]+)\}", tex)) == sorted(numbers)
    assert r"\newcommand{\val}" in tex


def test_a_key_is_defined_once_and_well_formed():
    k = bn.Keys()
    k["a.b"] = "1"
    with pytest.raises(ValueError, match="twice"):
        k["a.b"] = "2"
    with pytest.raises(ValueError, match="malformed"):
        k["A_b"] = "1"


def test_latex_types_a_minus_only_before_a_number():
    assert bn.latex("-0.1, -0.2") == r"\ensuremath{-}0.1, \ensuremath{-}0.2"
    assert bn.latex("DNABERT-2 +0.1") == "DNABERT-2 +0.1"


def test_a_flipped_name_is_marked_and_a_mixed_change_still_raises():
    assert bn.mark("NT-v2", ["GENA-LM", "NT-v2"]) == "NT-v2" + bt.MARK
    assert bn.mark("6", ["4"]) == "6" + bt.MARK
    assert bn.mark("0.660", ["0.661"]) == "0.660" + bt.MARK
    with pytest.raises(bt.StructureChanged):
        bn.mark("NT-v2 0.1", ["GENA-LM 0.1"])


def test_comments_are_stripped_after_an_even_run_of_backslashes():
    assert bn._strip_comments(r"a 5\% b") == r"a 5\% b"
    assert bn._strip_comments("a \\\\% gone \\pending{x}") == "a \\\\"


def test_counts_and_statistics_share_their_label_inputs(monkeypatch, tmp_path):
    c = json.loads((R.V2 / "counts.json").read_text())
    p = tmp_path / "counts.json"
    p.write_text(json.dumps({**c, "inputs": {**c["inputs"], "hgnc": "0" * 64}}))
    monkeypatch.setattr(bn, "COUNTS", p)
    with pytest.raises(R.MixedRecords, match="hgnc"):
        bn.build()


def test_the_margin_kept_under_homology_control_is_derived_from_its_keys(numbers):
    """Share of NT-v2's above-chance margin kept on the primary split, chance being
    the primary split's shuffled-label median (the random split has no null band)."""
    def v(k):
        return float(numbers[k].replace(bt.MARK, "").replace(r"\ensuremath{-}", "-"))
    for arm, null in (("cds", "null.hom.f5.kmer.median"), ("tss", "null.dis.f5.enformer-tss-4mer.median")):
        want = (v(f"{arm}.f5.enc.nt-v2") - v(null)) / (v(f"rand.{arm}.f5.enc.nt-v2") - v(null)) * 100
        assert abs(float(numbers[f"leak.{arm}.nt-v2.kept-pct"].replace(bt.MARK, "")) - want) < 1.0


def test_ridge_robust_rows_and_retrieval_chance_have_keys(numbers):
    rr = json.loads((R.V2 / "ridge_robust.json").read_text())
    esm = next(r for r in rr["rows"] if r["key"].endswith("/esm2_650m"))
    assert numbers["rr.esm2-650m.top5-pct"] == f"{esm['top5'] * 100:.1f}"
    assert numbers["rr.chance-top5-pct"] == f"{esm['chance_top5'] * 100:.1f}"


def test_a_bare_result_decimal_in_the_prose_is_reported():
    texts = {"a.tex": r"scores 0.693 here, \val{t1.delta} there, width=0.62\textwidth, 2.5--97.5\%"}
    assert bn.bare_decimals(texts) == [("a.tex", "0.693")]
