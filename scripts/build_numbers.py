#!/usr/bin/env python3
"""Every number the manuscript's prose states, written to
``dna_to_text_paper/paper/numbers.tex`` under a key named for what it measures.
The prose prints one as ``\\val{t1.delta}``; an undefined key stops the LaTeX
build, so no prose decimal is hand-transcribed (Phase 6).

Keys (``<split>`` is hom, dis or rand: the CDS primary, TSS primary and random
splits; ``<task>`` is f5 or gp; sources and labels are lower-case slugs):

  cell.<split>.<arm>.<task>.<source>     a cell's test macro-F1 or GenePT R^2
  <arm>.<task>.*, rand.<arm>.<task>.*     the validation picks: best encoder (name,
        pool, value), each encoder at its pool and minus the composition floor,
        the nucleotide and amino-acid k, ESM-2, Enformer
  t1-t4.{delta,ci,p,p-holm}               the four primary tests (Holm-adjusted together)
  ex.<split>.<task>.<label>.{delta,ci,p}  exploratory paired tests
  pool3x.*, masked.<mask>.*               the D5 pooling and masked-label tests (with the
        genes each mask removes: excluded, excluded-dis on the disjoint split)
  ``.p`` is always the unadjusted one-sided p; only ``.p-holm`` is adjusted.
  ci.*, null.*.{median,band}, chance.*.median   intervals, null bands, per-cell chance
  leak.<arm>.nt-v2.kept-pct              NT-v2's above-chance margin kept on the primary
        split, relative to the random split; chance is the primary split's
        shuffled-label median (the random split has no null band)
  rr.<source>.{macro,pooled,top5-pct,median-rank}, rr.chance-top5-pct, rr.n-test   ridge_robust.json
  pool.<enc>.spread, pool.between-encoders.spread   CDS family5 macro-F1 ranges
  scored.<split>.<arm>.{n,<family>}      scored test genes per family (family5, after the purge)
  t1-t4.ci-half                          half a primary test's interval width (2 dp)
  n.*, purge.*, pop.*, tss-overlap.*, single-chunk.*, selsens.*   counts; tss-overlap
        is the homology split's cross-partition window overlap over its test
        partition, and tss-overlap.scored.* over the test genes left after the purge
  n.family.<fam>, n.clusters, n.shared-cluster   family sizes; 40% protein clusters and the
        genes in a cluster of two or more (counts.json)
  edge.nonconverged                      probe cells whose pick sits next to a fit that
        did not converge, on the grid or in an extension (unshuffled records)
  seed.<cds|tss>.test-shared.{min,max}-pct   test genes any two of the four seed splits
        share, as a percentage of the primary split's test size (counts.json)
  n.translation.{irregular,internal-stop,internal-stop-olfactory,non-acgt}
        the irregular canonical CDS in data/translation_exceptions.tsv (counts.json)
  tss-pad.{n,min-bp,max-bp}              windows padded with N past a chromosome end, from
        the window manifest's pad_up + pad_down
  subm.<key>                             a value the submission printed (data/submitted_values.json,
        read verbatim from the submission source), for the changes-since-submission appendix
  tss-comp.*                             the TSS windows' annotation make-up (the target-CDS
        share, each partition bucket's percentage, genes, Ensembl release), from
        scripts/tss_overlap.py's audit table

Values are replayed through both protocol perturbations as the tables are
(``build_paper_tables.load_records``); a number whose digits move carries
``\\sens{}``. ``--check`` is the submission gate: it fails while any ``\\pending{}``
site or undefined ``\\val{}`` remains.

Run: uv run scripts/build_numbers.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys

import build_paper_tables as bt
from data_loader.enformer_windows import GTF_PATH, MANIFEST, window_spans
from data_loader.pool_names import POOL_DISPLAY
from linear_trainer import records as R
from splits.window_leak import window_leak_stats

PAPER = bt.ROOT / "dna_to_text_paper" / "paper"
OUT = PAPER / "numbers.tex"
COUNTS = R.V2 / "counts.json"
KEY = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
SPLITS = {"hom": bt.CDS, "dis": bt.TSS, "rand": bt.RAND}
TASKS = {"f5": "family5", "gp": "genept"}
AUDIT = R.REPO_ROOT / "analysis" / "tss_overlap" / "tables"
SUBMITTED = R.REPO_ROOT / "data" / "submitted_values.json"


def slug(s: str) -> str:
    s = s.lower().replace(">", " gt ").replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def latex(s: str) -> str:
    """A typeset minus for every negative number (a bare '-' prints a hyphen)."""
    return re.sub(r"(?<![\w{])-(?=\d)", r"\\ensuremath{-}", s)


def count(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _pair(lo: float, hi: float, dp: int = 3, signed: bool = False) -> str:
    fmt = bt.sgn if signed else bt.f
    return f"{fmt(lo, dp)}, {fmt(hi, dp)}"


class Keys(dict):
    def __setitem__(self, key: str, value: str):
        if not KEY.fullmatch(key):
            raise ValueError(f"malformed key {key!r}")
        if key in self:
            raise ValueError(f"key {key!r} defined twice")
        super().__setitem__(key, value)


def _paired(out: Keys, key: str, d: dict) -> None:
    out[f"{key}.delta"] = bt.sgn(d["delta_point"], 3)
    out[f"{key}.ci"] = _pair(*d["delta_ci95"], signed=True)
    out[f"{key}.p"] = f"{d['p_one_sided']:.3f}"
    if "p_holm" in d:
        out[f"{key}.p-holm"] = f"{d['p_holm']:.3f}"


def _task_key(task: str) -> str:
    return next(k for k, v in TASKS.items() if v == task)


def _split_key(split_file: str) -> str:
    return next(k for k, v in SPLITS.items() if v == split_file)


def _records(out: Keys) -> None:
    splits = {"hom": bt.HOM, "dis": bt.DIS, "rand": bt.RND}
    for sk, split in splits.items():
        for arm in ("cds", "tss"):
            for tk, task in TASKS.items():
                m = bt.F1 if task == "family5" else bt.R2
                for src, rec in split.cells(arm, task).items():
                    out[f"cell.{sk}.{arm}.{tk}.{slug(src)}"] = bt.f(rec[m], 3)
    for prefix, split, arm in (("cds", bt.HOM, "cds"), ("tss", bt.DIS, "tss"),
                               ("rand.cds", bt.RND, "cds"), ("rand.tss", bt.RND, "tss")):
        for tk, task in TASKS.items():
            m = bt.F1 if task == "family5" else bt.R2
            key = f"{prefix}.{tk}"
            if arm == "cds":
                floor_src = split.nt(task)
                out[f"{key}.nt-kmer.k"] = floor_src.removeprefix("kmer") or "4"
                out[f"{key}.nt-kmer"] = bt.f(split.cell(arm, task, floor_src)[m], 3)
                aa = split.aa(task)
                out[f"{key}.aa-kmer.k"] = aa.removeprefix("aa")
                out[f"{key}.aa-kmer"] = bt.f(split.cell(arm, task, aa)[m], 3)
                out[f"{key}.esm"] = bt.f(split.cell(arm, task, "esm2_650m")[m], 3)
                floor, below = split.cell(arm, task, floor_src)[m], "minus-nt"
            else:
                floor, below = split.cell(arm, task, bt.TSS_4MER)[m], "minus-4mer"
                out[f"{key}.4mer"] = bt.f(floor, 3)
                out[f"{key}.enformer"] = bt.f(split.cell(arm, task, bt.ENF_WHOLE)[m], 3)
                out[f"{key}.enformer-centre"] = bt.f(split.cell(arm, task, bt.ENF_CENTRE)[m], 3)
            best = split.best_encoder(arm, task)
            out[f"{key}.best-encoder"] = bt.ENC_DISPLAY[R.encoder_of(best)]
            out[f"{key}.best-encoder.pool"] = POOL_DISPLAY[bt.pool_of(best)]
            out[f"{key}.best-encoder.value"] = bt.f(split.cell(arm, task, best)[m], 3)
            for enc in bt.ENCODERS:
                rec = split.best(enc, arm, task)
                out[f"{key}.enc.{slug(enc)}"] = bt.f(rec[m], 3)
                out[f"{key}.enc.{slug(enc)}.pool"] = POOL_DISPLAY[bt.pool_of(rec["feature_source"])]
                out[f"{key}.enc.{slug(enc)}.{below}"] = bt.sgn(rec[m] - floor, 3)


def _statistics(out: Keys) -> None:
    s = bt.STATS
    for k, d in s["confirmatory"].items():
        _paired(out, k.split()[0].lower(), d)
        lo, hi = d["delta_ci95"]
        out[f"{k.split()[0].lower()}.ci-half"] = bt.f((hi - lo) / 2, 2)
    for k, d in s["exploratory"].items():
        split_file, rest = k.split(" ", 1)
        task, label = rest.split(": ", 1)
        _paired(out, f"ex.{_split_key(split_file)}.{_task_key(task)}.{slug(label)}", d)
    for cell, tests in s["pooling_3x"].items():
        task, enc = cell.split("/")
        for label, d in tests.items():
            if isinstance(d, dict) and "delta_point" in d:
                _paired(out, f"pool3x.{_task_key(task)}.{slug(enc)}.{slug(label)}", d)
    for name, block in s["sensitivity"].items():
        if name == "inputs":
            continue
        for label, d in block["tests"].items():
            first = label.split()[0]
            _paired(out, f"masked.{slug(name)}.{first.lower() if re.fullmatch(r'T\d', first) else slug(label)}", d)
        for suffix, on_dis in (("excluded", False), ("excluded-dis", True)):
            n = {d["n_excluded"] for k, d in block["tests"].items() if k.startswith("T4") is on_dis}
            if len(n) > 1:
                raise ValueError(f"{name}: the masked tests exclude different numbers of genes: {n}")
            if n:
                out[f"masked.{slug(name)}.{suffix}"] = count(n.pop())
    for section, key in (("intervals", "ci"), ("null_bands", "null"), ("chance", "chance")):
        for k, d in s[section].items():
            split_file, task, src = k.split("/")
            base = f"{key}.{_split_key(split_file)}.{_task_key(task)}.{slug(src)}"
            if section == "intervals":
                out[base] = _pair(*d["ci95"])
            elif section == "null_bands":
                out[f"{base}.median"] = bt.f(d["median"], 3)
                out[f"{base}.band"] = _pair(*d["band95"])
            else:
                out[f"{base}.median"] = bt.f(d["chance_median"], 3)


def _counts(out: Keys) -> None:
    c = json.loads(COUNTS.read_text())
    for name in (bt.CDS, bt.TSS):
        if c["inputs"][name] != R._sha(R.REPO_ROOT / "data" / name):
            raise R.MixedRecords(f"counts.json was built from another {name}; rerun scripts/build_counts.py")
    # The label counts here and the masked tests in statistics.json must read the same label inputs.
    for name, sha in bt.STATS["sensitivity"]["inputs"].items():
        if c["inputs"][name] != sha:
            raise R.MixedRecords(f"counts.json and statistics.json read different {name} files; rebuild both")
    out["n.genes"] = count(c["single_chunk"]["dnabert2"]["all genes"]["of"])
    for fam, n in c["families"].items():
        out[f"n.family.{slug(fam)}"] = count(n)
    if sum(c["families"].values()) != c["single_chunk"]["dnabert2"]["all genes"]["of"]:
        raise R.MixedRecords("counts.json: the family totals do not sum to the gene count")
    split_clusters = json.loads((R.REPO_ROOT / "data" / bt.CDS).read_text())["cluster_stats"]["n_clusters"]
    if c["clusters"]["n"] != split_clusters:
        raise R.MixedRecords(f"counts.json has {c['clusters']['n']} clusters, {bt.CDS} {split_clusters}")
    out["n.clusters"], out["n.shared-cluster"] = count(c["clusters"]["n"]), count(c["clusters"]["genes_in_shared"])
    tf, kin, tpl = c["noisy_tf_labels"], c["noisy_kinase_labels"], c["templated_summaries"]
    out["n.noisy-tf"], out["n.tf-labelled"] = count(tf["n"]["all genes"]), count(tf["of_tf_labelled"])
    out["n.noisy-tf.test"] = count(tf["n"]["splits.json test after the purge"])
    out["n.noisy-kinase"], out["n.kinase-labelled"] = count(kin["n"]["all genes"]), count(kin["of_kinase_labelled"])
    out["n.noisy-kinase.not-kinase"] = count(kin["not_kinase"]["all genes"])
    out["n.noisy-kinase.small-molecule"] = count(kin["small_molecule"]["all genes"])
    out["n.template"], out["n.template.groups"] = count(tpl["n"]["all genes"]), count(tpl["n_groups"])
    out["n.template.largest"] = count(tpl["largest_group"]["n"])
    out["n.template.empty"] = count(tpl["empty_summary"]["n"]["all genes"])
    for enc, d in c["single_chunk"].items():
        if isinstance(d, dict) and "all genes" in d:
            out[f"single-chunk.{slug(enc)}.pct"] = f"{d['all genes']['share'] * 100:.1f}"
            out[f"single-chunk.{slug(enc)}.test-pct"] = f"{d['splits.json test after the purge']['share'] * 100:.1f}"
    for name, sha in c["inputs"].items():       # the seed splits and the translation list behind these keys
        if (name.startswith("splits") or name == "translation_exceptions.tsv") \
                and sha != R._sha(R.REPO_ROOT / "data" / name):
            raise R.MixedRecords(f"counts.json was built from another {name}; rerun scripts/build_counts.py")
    for arm, d in c["seed_test_shared"].items():
        pct = [100 * k / d["n_primary_test"] for k in d["shared"].values()]
        out[f"seed.{arm}.test-shared.min-pct"] = f"{min(pct):.0f}"
        out[f"seed.{arm}.test-shared.max-pct"] = f"{max(pct):.0f}"
    t = c["translation_exceptions"]
    out["n.translation.irregular"], out["n.translation.internal-stop"] = count(t["n"]), count(t["internal_stop"])
    out["n.translation.internal-stop-olfactory"] = count(t["internal_stop_olfactory"])
    out["n.translation.non-acgt"] = count(t["non_acgt"])
    for sk in ("hom", "dis"):
        for part, d in c["split_population"][SPLITS[sk]].items():
            if part in ("train", "val", "test"):
                k = f"pop.{sk}.{part}"
                out[f"{k}.n"], out[f"{k}.olfactory"], out[f"{k}.gpcr"] = count(d["n"]), count(d["olfactory"]), count(d["gpcr"])
                out[f"{k}.singleton-pct"] = f"{d['singleton_share'] * 100:.0f}"
    # The evaluation purge, from the records themselves (one purge per split and arm).
    for sk, split, arm in (("cds", bt.HOM, "cds"), ("dis-cds", bt.DIS, "cds"), ("dis-tss", bt.DIS, "tss")):
        purges = {json.dumps(r["purge"], sort_keys=True) for r in split.recs.values() if r["arm"] == arm}
        if len(purges) != 1:
            raise R.MixedRecords(f"{split.name}/{arm}: {len(purges)} different purges")
        purge = json.loads(purges.pop())
        out[f"purge.{sk}.test"], out[f"purge.{sk}.val"] = count(len(purge["test_masked"])), count(len(purge["val_masked"]))
    # Scored test genes per family (after the purge): what each macro-F1 averages over.
    for sk, split, arm in (("hom", bt.HOM, "cds"), ("dis", bt.DIS, "tss")):
        by_class = {json.dumps(r["n_test_scored_by_class"], sort_keys=True) for r in split.recs.values()
                    if r["arm"] == arm and r["task"] == "family5"}
        if len(by_class) != 1:
            raise R.MixedRecords(f"{split.name}/{arm}: {len(by_class)} different scored test sets")
        fams = json.loads(by_class.pop())
        out[f"scored.{sk}.{arm}.n"] = count(sum(fams.values()))
        for fam, n in fams.items():
            out[f"scored.{sk}.{arm}.{slug(fam)}"] = count(n)
    _window_overlap(out)
    _window_padding(out)
    _window_composition(out)
    _submitted(out)


def _submitted(out: Keys) -> None:
    """The submission's printed values, for the changes-since-submission appendix."""
    for key, d in json.loads(SUBMITTED.read_text())["values"].items():
        out[f"subm.{key}"] = d["value"]


def _window_padding(out: Keys) -> None:
    """Windows that run past a chromosome end are padded with N (the same input for
    every model). Read from the manifest _window_overlap checked against the records."""
    with MANIFEST.open() as fh:
        pads = [int(r["pad_up"]) + int(r["pad_down"]) for r in csv.DictReader(fh, delimiter="\t")]
    padded = [p for p in pads if p > 0]
    out["tss-pad.n"] = count(len(padded))
    out["tss-pad.min-bp"], out["tss-pad.max-bp"] = count(min(padded)), count(max(padded))


def _window_composition(out: Keys) -> None:
    """How each TSS window divides into the gene's own CDS, UTRs and introns,
    neighbouring genes and intergenic sequence, from the audit table
    scripts/tss_overlap.py writes (painting the GTF takes minutes). The audit
    must have read the current window manifest, the GTF that manifest was built
    from (its tracked pin, so the gitignored GTF isn't needed here) and every gene."""
    prov = json.loads((AUDIT / "provenance.json").read_text())
    meta = json.loads(MANIFEST.with_suffix(".meta.json").read_text())
    if (prov["manifest_sha256"], prov["gtf_sha256"]) != (R._sha(MANIFEST), meta["inputs"][GTF_PATH.name]):
        raise R.MixedRecords("the TSS composition audit predates the current window manifest or GTF: "
                             "rerun scripts/tss_overlap.py")
    with (AUDIT / "overlap_by_family.csv").open() as fh:
        overall = next(r for r in csv.DictReader(fh) if r["group"] == "overall")
    if int(overall["n_genes"]) != meta["n_genes"]:
        raise R.MixedRecords(f"the TSS composition audit covers {overall['n_genes']} genes, "
                             f"the window manifest {meta['n_genes']}")
    out["tss-comp.target-cds"] = bt.f(float(overall["target_cds"]), 3)
    for bucket in prov["partition"]:
        out[f"tss-comp.{slug(bucket)}-pct"] = f"{float(overall[bucket]) * 100:.0f}"
    out["tss-comp.n"], out["tss-comp.ensembl"] = count(int(overall["n_genes"])), str(prov["ensembl_release"])


def _window_overlap(out: Keys) -> None:
    """TSS-window overlap across the homology split's partitions, computed here from
    the current split and window manifest (window_leak.json, the split builder's
    record, carries no split digest). The manifest must be the one the records'
    window purge read."""
    windows = {r["purge"]["windows_sha256"] for r in bt.DIS.recs.values() if "windows_sha256" in r["purge"]}
    if windows != {R._sha(MANIFEST)}:
        raise R.MixedRecords(f"the window manifest {MANIFEST.name} is not the one the records' window purge read")
    split = json.loads((R.REPO_ROOT / "data" / bt.CDS).read_text())
    spans = window_spans(MANIFEST)
    masked = {json.dumps(sorted(r["purge"]["test_masked"])) for r in bt.HOM.recs.values() if r["arm"] == "tss"}
    if len(masked) != 1:
        raise R.MixedRecords(f"{bt.CDS}/tss: {len(masked)} different purges")
    purged = set(json.loads(masked.pop()))
    scored = {**split, "test": [g for g in split["test"] if g not in purged]}
    for key, part in (("tss-overlap", split), ("tss-overlap.scored", scored)):
        leak = window_leak_stats(part, spans)
        out[f"{key}.test"], out[f"{key}.n-test"] = count(leak["test_overlapping_trainval"]), count(leak["n_test"])
        out[f"{key}.pct"] = f"{leak['frac_test_overlapping_trainval'] * 100:.1f}"
        out[f"{key}.pairs"] = count(leak["cross_split_pairs"])


def _selection_sensitivity(out: Keys) -> None:
    sens = json.loads(bt.SENS.read_text())
    R.check_inputs(sens)
    shuffled = re.compile(r"/shuf\d+$")
    main = [c for c in sens["cells"] if not shuffled.search(c["key"])]
    recs = [r for p in R.V2.glob("metrics_*.json") for r in json.loads(p.read_text()) if not r["shuffled_labels"]]
    out["selsens.main-cells"], out["selsens.moved"] = count(len(recs)), count(len(main))
    out["edge.nonconverged"] = count(sum(r["edge"] == "nonconverged" for r in recs))
    out["selsens.pick-changed"] = count(sum(c["pick_changed"] for c in main))
    out["selsens.max-df1"] = bt.f(max(c["max_abs_d_test_f1"] or 0.0 for c in main), 3)
    out["selsens.moved-over-001"] = count(sum((c["max_abs_d_test_f1"] or 0.0) > 0.01 for c in main))
    out["selsens.null-moved"] = count(len(sens["cells"]) - len(main))
    out["selsens.builder-picks"] = count(sens["summary"]["builder_picks"])


def _derived(out: Keys) -> None:
    """Quantities the prose states that combine records and statistics."""
    nulls = {"cds": bt.band(bt.CDS, "family5", "kmer")["median"],
             "tss": bt.band(bt.TSS, "family5", bt.TSS_4MER)["median"]}
    for arm, chance in nulls.items():
        prim, rand = bt.primary(arm).best("nt_v2", arm, "family5")[bt.F1], bt.RND.best("nt_v2", arm, "family5")[bt.F1]
        out[f"leak.{arm}.nt-v2.kept-pct"] = f"{(prim - chance) / (rand - chance) * 100:.0f}"
    rr = json.loads((R.V2 / "ridge_robust.json").read_text())
    R.check_inputs(rr)
    if rr["stamp"] != bt.STATS["stamp"]:
        raise R.MixedRecords("ridge_robust.json was rescored from other records")
    chance = {f"{r['chance_top5'] * 100:.1f}" for r in rr["rows"] if r["key"].startswith(bt.CDS)}
    if len(chance) != 1:
        raise ValueError(f"the CDS rows of ridge_robust.json disagree on retrieval chance: {chance}")
    out["rr.chance-top5-pct"] = chance.pop()
    n_test = {r["n_test"] for r in rr["rows"] if r["key"].startswith(bt.CDS)}
    if len(n_test) != 1:
        raise ValueError(f"the CDS rows of ridge_robust.json score different test sets: {n_test}")
    out["rr.n-test"] = count(n_test.pop())
    # Pooling spread: each encoder's range over its CDS rules, against the range across encoders at their picks.
    by = bt.HOM.cells("cds", "family5")
    for enc in bt.ENCODERS:
        vals = [by[f"{enc}_{p}"][bt.F1] for p in bt.encoder_pools(enc, "CDS")]
        out[f"pool.{slug(enc)}.spread"] = bt.f(max(vals) - min(vals), 3)
    picks = [bt.HOM.best(e, "cds", "family5")[bt.F1] for e in bt.ENCODERS]
    out["pool.between-encoders.spread"] = bt.f(max(picks) - min(picks), 3)
    for r in rr["rows"]:
        if r["key"].startswith(f"{bt.CDS}/cds/genept/"):
            k = f"rr.{slug(r['key'].rsplit('/', 1)[1])}"
            out[f"{k}.macro"], out[f"{k}.pooled"] = bt.f(r["macro_r2"], 3), bt.f(r["pooled_r2"], 3)
            out[f"{k}.top5-pct"], out[f"{k}.median-rank"] = f"{r['top5'] * 100:.1f}", f"{r['median_rank']:.0f}"


def collect() -> Keys:
    out = Keys()
    _records(out)
    _statistics(out)
    _derived(out)
    _counts(out)
    _selection_sensitivity(out)
    return out


def mark(value: str, variants: list[str]) -> str:
    """MARK the digits that move; a name or k (no decimals) that flips is marked whole."""
    if not bt.NUM.search(value) and any(v != value for v in variants):
        if any(bt.NUM.search(v) for v in variants):
            raise bt.StructureChanged(f"{value!r} becomes a number in a replay")
        return value + bt.MARK
    return bt.mark_unstable(value, variants)


def build() -> dict[str, str]:
    """Every key, from the canonical records, marked where a replay moves it."""
    variants = []
    for p in bt.PERTURBATIONS:
        bt.load_records(p)
        variants.append(collect())
    bt.load_records()
    canon = collect()
    return {k: latex(mark(v, [var[k] for var in variants])) for k, v in canon.items()}


def render(numbers: dict[str, str]) -> str:
    lines = [
        "% Generated by scripts/build_numbers.py from data/v2; do not edit. The prose prints a",
        "% value as \\val{key}; a dagger (\\sens) marks digits that move under the 6-thread or",
        "% other-kernel refit of the frozen protocol.",
        r"\makeatletter",
        r"\newcommand{\val}[1]{\@ifundefined{mina@#1}{\PackageError{numbers}{No value named #1}"
        r"{Fix the key or rerun scripts/build_numbers.py}}{\@nameuse{mina@#1}}}",
        *(rf"\@namedef{{mina@{k}}}{{{v}}}" for k, v in numbers.items()),
        r"\makeatother",
    ]
    return "\n".join(lines) + "\n"


def _strip_comments(text: str) -> str:
    """Drop TeX comments: a % after an even run of backslashes (\\% is a percent sign)."""
    return re.sub(r"(?<!\\)((?:\\\\)*)%.*", r"\1", text)


PROSE = {f"{n}.tex" for n in ("abstract", "introduction", "methods", "results", "conclusion", "appendix")}


def paper_sources() -> dict[str, str]:
    return {p.name: _strip_comments(p.read_text()) for p in sorted(PAPER.glob("*.tex")) if p != OUT}


def undefined(texts: dict[str, str], keys: set[str]) -> list[tuple[str, str]]:
    return [(name, k) for name, t in texts.items() for k in re.findall(r"\\val\{([^}]*)\}", t) if k not in keys]


def pending(texts: dict[str, str]) -> list[tuple[str, str]]:
    return [(name, v) for name, t in texts.items() if name != "header.tex"
            for v in re.findall(r"\\pending\{([^}]*)\}", t)]


# Decimals in the prose that are not results: layout, model sizes, tool flags,
# fixed facts. Anything else must come through \val (G: no hand-transcribed number).
NOT_RESULTS = re.compile(r"width=\d|\d\\(column|text)width|\d\{\\,\}M|--min-seq-id|min_dist=|GRCh38\.|"
                         r"\$\\geq\$\d|2\.5--97\.5|99\.9\\%|\d kb| -c \d|10\.5281/zenodo|CC BY(?:-NC-SA)? \d\.\d")


def bare_decimals(texts: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    for name, t in texts.items():
        t = re.sub(r"\\(val|pending)\{[^}]*\}", "", t)
        for m in re.finditer(r"\d+\.\d+", t):
            around = t[max(0, m.start() - 15):m.end() + 15]
            if not NOT_RESULTS.search(around):
                out.append((name, m.group()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail while a \\pending{} or undefined \\val{} remains")
    args = ap.parse_args()
    numbers = build()
    OUT.write_text(render(numbers))
    texts = paper_sources()
    missing, left = undefined(texts, set(numbers)), pending(texts)
    bare = bare_decimals({k: v for k, v in texts.items() if k in PROSE})
    print(f"wrote {OUT.name}: {len(numbers)} keys, {sum(bt.MARK in v for v in numbers.values())} marked; "
          f"{len(left)} \\pending sites left, {len(bare)} bare decimals, {len(missing)} undefined \\val")
    for name, k in missing:
        print(f"  undefined: {name}: \\val{{{k}}}")
    for name, v in bare if args.check else ():
        print(f"  bare decimal: {name}: {v}")
    if missing or (args.check and (left or bare)):
        sys.exit(1)


if __name__ == "__main__":
    main()
