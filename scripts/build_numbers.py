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
  t1-t4.{delta,ci,p,p-holm}               the confirmatory tests
  ex.<split>.<task>.<label>.{delta,ci,p}  exploratory paired tests
  pool3x.*, masked.<mask>.*               the D5 pooling and masked-label tests (with the
        genes each mask removes: excluded, excluded-dis on the disjoint split)
  ``.p`` is always the unadjusted one-sided p; only ``.p-holm`` is adjusted.
  ci.*, null.*.{median,band}, chance.*.median   intervals, null bands, per-cell chance
  n.*, purge.*, pop.*, tss-overlap.*, single-chunk.*, selsens.*   counts

Values are replayed through both protocol perturbations as the tables are
(``build_paper_tables.load_records``); a number whose digits move carries
``\\sens{}``. ``--check`` is the submission gate: it fails while any ``\\pending{}``
site or undefined ``\\val{}`` remains.

Run: uv run scripts/build_numbers.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import build_paper_tables as bt
from data_loader.pool_names import POOL_DISPLAY
from linear_trainer import records as R

PAPER = bt.ROOT / "dna_to_text_paper" / "paper"
OUT = PAPER / "numbers.tex"
WINDOW_LEAK = R.REPO_ROOT / "analysis" / "tss_overlap" / "window_leak.json"
COUNTS = R.V2 / "counts.json"
KEY = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
SPLITS = {"hom": bt.CDS, "dis": bt.TSS, "rand": bt.RAND}
TASKS = {"f5": "family5", "gp": "genept"}


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
    leak_all = json.loads(WINDOW_LEAK.read_text())
    if leak_all["window_manifest_sha256"] != R._sha(R.REPO_ROOT / "data" / "tss_windows.tsv"):
        raise R.MixedRecords("window_leak.json was measured on other TSS windows than data/tss_windows.tsv")
    leak = leak_all[bt.CDS]
    out["tss-overlap.test"], out["tss-overlap.n-test"] = count(leak["test_overlapping_trainval"]), count(leak["n_test"])
    out["tss-overlap.pct"] = f"{leak['frac_test_overlapping_trainval'] * 100:.1f}"
    out["tss-overlap.pairs"] = count(leak["cross_split_pairs"])


def _selection_sensitivity(out: Keys) -> None:
    sens = json.loads(bt.SENS.read_text())
    R.check_inputs(sens)
    shuffled = re.compile(r"/shuf\d+$")
    main = [c for c in sens["cells"] if not shuffled.search(c["key"])]
    n_main = sum(not r["shuffled_labels"] for p in R.V2.glob("metrics_*.json") for r in json.loads(p.read_text()))
    out["selsens.main-cells"], out["selsens.moved"] = count(n_main), count(len(main))
    out["selsens.pick-changed"] = count(sum(c["pick_changed"] for c in main))
    out["selsens.max-df1"] = bt.f(max(c["max_abs_d_test_f1"] or 0.0 for c in main), 3)
    out["selsens.moved-over-001"] = count(sum((c["max_abs_d_test_f1"] or 0.0) > 0.01 for c in main))
    out["selsens.null-moved"] = count(len(sens["cells"]) - len(main))
    out["selsens.builder-picks"] = count(sens["summary"]["builder_picks"])


def collect() -> Keys:
    out = Keys()
    _records(out)
    _statistics(out)
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


def paper_sources() -> dict[str, str]:
    return {p.name: _strip_comments(p.read_text()) for p in sorted(PAPER.glob("*.tex")) if p != OUT}


def undefined(texts: dict[str, str], keys: set[str]) -> list[tuple[str, str]]:
    return [(name, k) for name, t in texts.items() for k in re.findall(r"\\val\{([^}]*)\}", t) if k not in keys]


def pending(texts: dict[str, str]) -> list[tuple[str, str]]:
    return [(name, v) for name, t in texts.items() if name != "header.tex"
            for v in re.findall(r"\\pending\{([^}]*)\}", t)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail while a \\pending{} or undefined \\val{} remains")
    args = ap.parse_args()
    numbers = build()
    OUT.write_text(render(numbers))
    texts = paper_sources()
    missing, left = undefined(texts, set(numbers)), pending(texts)
    print(f"wrote {OUT.name}: {len(numbers)} keys, {sum(bt.MARK in v for v in numbers.values())} marked; "
          f"{len(left)} \\pending sites left, {len(missing)} undefined \\val")
    for name, k in missing:
        print(f"  undefined: {name}: \\val{{{k}}}")
    if missing or (args.check and left):
        sys.exit(1)


if __name__ == "__main__":
    main()
