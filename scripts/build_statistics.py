"""Every interval, paired test and null band the paper reports (Phase 1D, D4).

Reads the camera-ready records (``data/v2``, via ``linear_trainer.records``),
picks each headline cell on validation scores only, and runs the statistics in
``linear_trainer.stats`` on their stored predictions. Writes
``data/v2/statistics.json``. Refuses to build without ``run_complete.json``
(one whole-manifest run) and a passing ``reproduction.json`` for the same
records (``scripts/reproduce_headline.py``, Rule 3), and cross-checks T1-T4
against that independent reimplementation.

Confirmatory family (Holm-adjusted together; decided Sept 29), family5 macro-F1,
each side the validation-selected cell, one-sided for A > B:

  T1  best DNA encoder (CDS)  vs  nucleotide k-mer (k in {4, 6} chosen on val)
  T2  best DNA encoder (CDS)  vs  amino-acid k-mer (k in {1, 2, 3} chosen on val)
  T3  ESM-2 650M              vs  best DNA encoder (CDS)
  T4  best encoder on CDS     vs  the same encoder on TSS (both on the disjoint split)

T1-T3 run on the CDS primary split (``splits.json``); T4 on the TSS primary
(``splits_tss_disjoint.json``), where the CDS cells also run, so both sides
score the same test genes. Everything else is exploratory and unadjusted,
including two D5 blocks:

  sensitivity  T1-T4 with the noisy TF labels, and separately the kinase-labelled
               genes that are not protein kinases, excluded from scoring; the
               GenePT comparisons with the templated-summary genes excluded
               (data_loader.label_audit); masks on stored predictions, no refit
  pooling_3x   Ends + Mean against Mean copied three times (= Mean at 3x C, the
               ``<enc>_meanmean3`` control) and against Mean on its own grid

and four from the Rule 3 audit (Oct 1):

  exploratory  also "encoder > composition + CDS length" (``<k-mer>_len``
               controls; the encoders see length through their chunking)
  chance       every interval cell's own chance level: its stored predictions
               against permuted test labels
  within_family  GenePT R^2 with each family's mean target removed
  seed_replication  T1-T3 on the re-seeded CDS splits, unadjusted

Run: uv run scripts/build_statistics.py [--n-iters 1000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_loader import label_audit
from linear_trainer import records as R
from linear_trainer import stats
from linear_trainer.cell import scored_predictions

CDS = "splits.json"
TSS = "splits_tss_disjoint.json"
NULL_SHUFFLES = 200
SEEDS = (1, 7, 123)
OUT = R.V2 / "statistics.json"


def _headline(recs: dict, task: str) -> dict[str, str]:
    cds = R.cells(recs, "cds", task)
    return {"best_encoder": R.best_encoder(cds, "cds"), "nt_kmer": R.best_nt_kmer(cds),
            "aa_kmer": R.best_aa(cds), "esm2": "esm2_650m",
            **{f"best_{e}": R.best_pool(cds, e, "cds") for e in R.ENCODERS}}


# A confirmatory cell whose pick sits at a search limit, stopped on a fit that
# didn't converge, whose train+val refit didn't converge, that predicts one
# class, or that scores fewer than MIN_PER_FAMILY test genes of some family
# (each family is a fifth of macro-F1; G28) fails the build. "plateau" (the
# next decade ties within plateau_eps) is a stable pick and is reported, not
# refused.
BAD_EDGES = ("limit", "nonconverged")
MIN_PER_FAMILY = 15


class UnsoundConfirmatoryCell(RuntimeError):
    """A confirmatory test would rest on an edge or degenerate pick."""


def check_confirmatory_cell(rec: dict) -> None:
    if rec["edge"] in BAD_EDGES or rec["degenerate"] or not rec["converged"]:
        raise UnsoundConfirmatoryCell(f"{rec['key']}: edge={rec['edge']!r}, "
                                      f"degenerate={rec['degenerate']!r}, "
                                      f"refit converged={rec['converged']!r}")
    thin = {c: n for c, n in rec["n_test_scored_by_class"].items() if n < MIN_PER_FAMILY}
    if len(rec["n_test_scored_by_class"]) != 5 or thin:
        raise UnsoundConfirmatoryCell(f"{rec['key']}: scored test genes per family "
                                      f"{rec['n_test_scored_by_class']} (floor {MIN_PER_FAMILY})")


def cds_pairs(cds_recs: dict) -> dict[str, tuple[dict, dict]]:
    """T1-T3 on one CDS split file, each side picked on that file's validation."""
    cds = R.cells(cds_recs, "cds", "family5")
    h = _headline(cds_recs, "family5")
    enc = h["best_encoder"]
    return {
        "T1 encoder > nucleotide k-mer": (cds[enc], cds[h["nt_kmer"]]),
        "T2 encoder > amino-acid k-mer": (cds[enc], cds[h["aa_kmer"]]),
        "T3 ESM-2 650M > encoder": (cds["esm2_650m"], cds[enc]),
    }


def confirmatory_pairs(cds_recs: dict, tss_recs: dict) -> dict[str, tuple[dict, dict]]:
    tests = cds_pairs(cds_recs)
    # T4 on the disjoint split: the encoder picked on its own CDS validation there,
    # against that encoder's own TSS pick.
    dcds = R.cells(tss_recs, "cds", "family5")
    dtss = R.cells(tss_recs, "tss", "family5")
    enc_d = R.best_encoder(dcds, "cds")     # may differ from T1-T3's; the output names both
    tests["T4 CDS > TSS (same encoder)"] = (dcds[enc_d], dtss[R.best_pool(dtss, R.encoder_of(enc_d), "tss")])
    return tests


def confirmatory(cds_recs: dict, tss_recs: dict, n_iters: int) -> dict:
    tests = confirmatory_pairs(cds_recs, tss_recs)
    for a, b in tests.values():
        check_confirmatory_cell(a)
        check_confirmatory_cell(b)
    out = {k: stats.paired_bootstrap(a, b, n_iters=n_iters) for k, (a, b) in tests.items()}
    for k, (a, b) in tests.items():
        out[k]["edges"] = [a["edge"], b["edge"]]
    # T1-T3 must use one encoder cell; T4 names its own (picked on the disjoint split).
    adj = stats.holm({k: v["p_one_sided"] for k, v in out.items()})
    for k in out:
        out[k]["p_holm"] = adj[k]
    return out


# How far the independent bootstrap (its own seed) may sit from ours: Monte Carlo
# error at 1,000 resamples is about 0.005 on a CI endpoint and 0.01 on p.
REPRO_CI_TOL, REPRO_P_TOL = 0.02, 0.03


def check_reproduction(conf: dict, theirs: dict) -> None:
    """T1-T4 must agree with the independent reimplementation (Rule 3): the same
    two cells, the same point, test genes and resampling groups, and an interval
    and p within Monte Carlo tolerance of its own bootstrap."""
    for label, ours in conf.items():
        t = label.split()[0]
        other = theirs.get(t)
        if other is None:
            raise R.NotReproduced(f"{t}: the reproduction has no result for it")
        bad = [k for k in ("a", "b", "n_test", "n_groups") if ours[k] != other[k]]
        if abs(ours["delta_point"] - other["delta_point"]) > 1e-12:
            bad.append("delta_point")
        if max(abs(x - y) for x, y in zip(ours["delta_ci95"], other["delta_ci95"])) > REPRO_CI_TOL:
            bad.append("delta_ci95")
        if abs(ours["p_one_sided"] - other["p_one_sided"]) > REPRO_P_TOL:
            bad.append("p_one_sided")
        if bad:
            raise R.NotReproduced(f"{t} disagrees with the independent reproduction on {bad}: "
                                 f"{ {k: (ours[k], other[k]) for k in bad} }")


def seed_replication(seed_recs: dict[str, dict], n_iters: int) -> dict:
    """T1-T3 re-run on each re-seeded CDS split, unadjusted (Rule 3: the tests were
    fixed after the May numbers were seen). The seeds re-deal only the small
    clusters (the largest stay in train), so this is a partial replication."""
    out = {}
    for split, recs in seed_recs.items():
        out[split] = {}
        for k, (a, b) in cds_pairs(recs).items():
            out[split][k] = {**stats.paired_bootstrap(a, b, n_iters=n_iters), "edges": [a["edge"], b["edge"]]}
    return out


def within_family(recs: dict[str, dict], n_iters: int) -> dict:
    """Within-family GenePT R^2 for the CDS-arm GenePT interval cells on the CDS
    primary (Rule 3, keyed as in ``intervals``): family means from the fit rows
    (train + val) of the split file."""
    from linear_trainer import sources
    from splits import load_split
    split_path = R.REPO_ROOT / "data" / CDS
    family: dict[str, str] = {}
    fit_y, fit_f = [], []
    for part in ("train", "val", "test"):
        _, Y, meta = load_split(part, dataset_path=sources.META_PARQUET, splits_path=split_path)
        family.update(zip(meta["ensembl_id"], meta["family"]))
        if part != "test":
            fit_y.append(Y)
            fit_f.append(meta["family"].to_numpy())
    Y, fam = np.vstack(fit_y).astype(np.float64), np.concatenate(fit_f)
    means = {f: Y[fam == f].mean(axis=0) for f in sorted(set(fam))}
    if any((r["split"], r["arm"], r["task"]) != (CDS, "cds", "genept") for r in recs.values()):
        raise ValueError("within-family R^2 is computed for CDS-arm GenePT cells on the CDS primary only")
    return {k: stats.within_family_r2(rec, family, means, n_iters=n_iters) for k, rec in sorted(recs.items())}


def exploratory_pairs(recs: dict, arm_pairs: list[tuple[str, str, str, str]], n_iters: int) -> dict:
    out = {}
    for label, task, a, b in arm_pairs:
        cells_ = {**R.cells(recs, "cds", task), **R.cells(recs, "tss", task)}
        missing = [x for x in (a, b) if x not in cells_]
        if missing:
            raise R.MissingRecord(f"{label}: no record for {missing}")
        out[label] = stats.paired_bootstrap(cells_[a], cells_[b], n_iters=n_iters)
    return out


def _genept_pairs(recs: dict) -> dict[str, tuple[dict, dict]]:
    cds = R.cells(recs, "cds", "genept")
    h = _headline(recs, "genept")
    enc = h["best_encoder"]
    return {"genept: encoder > nt_kmer": (cds[enc], cds[h["nt_kmer"]]),
            "genept: encoder > aa_kmer": (cds[enc], cds[h["aa_kmer"]]),
            "genept: esm2_650m > encoder": (cds["esm2_650m"], cds[enc])}


def _check_noisy_labels(recs: list[dict], noisy: frozenset[str], label: str) -> None:
    """The mask is built from the Stage 1 gene table, not from the records: every
    masked gene a record scores must carry ``label`` in its stored predictions,
    or the gene table has drifted from the parquets the records were fitted on."""
    for rec in recs:
        arrays = scored_predictions(rec)
        hit = np.isin(arrays["ids"], sorted(noisy))
        wrong = sorted(set(arrays["y_true"][hit].tolist()) - {label})
        if wrong:
            raise R.MixedRecords(f"{rec['key']}: noisy-{label} mask hits genes labelled {wrong}")


def sensitivity(cds_recs: dict, tss_recs: dict, n_iters: int) -> dict:
    """The headline tests with each disclosed defect masked from scoring (D5)."""
    gene_table, hgnc, inputs = label_audit.load_inputs()
    noisy = label_audit.noisy_tf_genes(gene_table, hgnc)
    kinase = frozenset().union(*label_audit.noisy_kinase_genes(gene_table, hgnc).values())
    pairs = confirmatory_pairs(cds_recs, tss_recs)
    sides = [r for ab in pairs.values() for r in ab]
    _check_noisy_labels(sides, noisy, "tf")
    _check_noisy_labels(sides, kinase, "kinase")
    masks = {"label_noise": ("family5", noisy, pairs),
             "label_noise_kinase": ("family5", kinase, pairs),       # Rule 3 (G27)
             "template": ("genept", label_audit.templated_genes(gene_table), _genept_pairs(cds_recs))}
    out: dict = {"inputs": inputs}
    for name, (task, genes, pairs) in masks.items():
        h = _headline(cds_recs, task)
        cds = R.cells(cds_recs, "cds", task)
        heads = sorted({h["best_encoder"], h["nt_kmer"], h["aa_kmer"], h["esm2"]})
        out[name] = {
            "task": task, "n_genes": len(genes),
            "tests": {k: stats.paired_bootstrap(a, b, n_iters=n_iters, exclude=genes)
                      for k, (a, b) in pairs.items()},
            "intervals": {f"{CDS}/{task}/{src}": stats.cluster_bootstrap(cds[src], n_iters=n_iters,
                                                                         exclude=genes)
                          for src in heads},
        }
    return out


def pooling_3x(cds_recs: dict, n_iters: int) -> dict:
    """Ends + Mean (meanD) against the budget-matched Mean (D5, the 3x C test).

    ``<enc>_meanmean3`` is Mean on a 3x-shifted grid, and equals Ends + Mean for
    single-chunk genes, so the first pair isolates chunk position; the second is
    the comparison the pooling table shows. Exploratory, CDS primary."""
    out: dict = {}
    for task in ("family5", "genept"):
        cds = R.cells(cds_recs, "cds", task)
        hp = "C" if task == "family5" else "alpha"
        for e in R.ENCODERS:
            ends, mean, mean3 = (cds[f"{e}_{p}"] for p in ("meanD", "meanmean", "meanmean3"))
            row = {"Ends+Mean > Mean x3 (3x C)": stats.paired_bootstrap(ends, mean3, n_iters=n_iters),
                   "Ends+Mean > Mean": stats.paired_bootstrap(ends, mean, n_iters=n_iters),
                   "picks": {"meanD": ends[hp], "meanmean": mean[hp], "meanmean3": mean3[hp]},
                   "edges": {"meanD": ends["edge"], "meanmean": mean["edge"], "meanmean3": mean3["edge"]}}
            if task == "family5":   # the share of scored test genes on which the two agree
                a, b = scored_predictions(ends), scored_predictions(mean3)
                pos = {g: i for i, g in enumerate(b["ids"].tolist())}
                if set(pos) != set(a["ids"].tolist()):
                    raise ValueError(f"{e}: meanD and meanmean3 score different test genes")
                order = np.array([pos[g] for g in a["ids"].tolist()])
                row["agreement_with_mean_x3"] = float(np.mean(a["pred"] == b["pred"][order]))
            out[f"{task}/{e}"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iters", type=int, default=stats.N_ITERS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    n = args.n_iters

    cds_recs, tss_recs = R.load(CDS), R.load(TSS)
    nulls = {split: R.load(split, null=True) for split in (CDS, TSS)}
    seeds = {f"splits_seed{s}.json": R.load(f"splits_seed{s}.json") for s in SEEDS}
    stamp = R.stamp_of(cds_recs, tss_recs, *nulls.values(), *seeds.values())   # G7 across every file used
    R.check_complete(stamp)                                           # G17, G1: one whole-manifest run
    reproduced = R.check_reproduced()                                 # Rule 3: independent refit matched
    controls = ([f"{e}_meanmean3" for e in R.ENCODERS]                # D5: added after the 746-cell runs
                + [*R.NT_KMERS_LEN, *R.AA_KMERS_LEN])                 # Rule 3: composition + length
    missing = [f"{t}/{c}" for t in ("family5", "genept") for c in controls
               if c not in R.cells(cds_recs, "cds", t)]
    if missing:
        raise R.MissingRecord(f"no record for the controls {missing}; these records predate "
                              "them, so rerun the whole manifest")
    result: dict = {"stamp": stamp, "inputs": R.input_digests(), "n_iters": n, "seed": stats.SEED,
                    "confirmatory": confirmatory(cds_recs, tss_recs, n)}
    check_reproduction(result["confirmatory"], reproduced["statistics"])

    interval_recs: dict[str, dict] = {}

    def ci(recs, arm, task, src):
        rec = R.cells(recs, arm, task)[src]
        interval_recs[f"{rec['split']}/{task}/{src}"] = rec
        return stats.cluster_bootstrap(rec, n_iters=n)

    # Intervals: every CDS comparator and encoder pick on the CDS primary; every
    # TSS pick, Enformer readout and E5 cell on the TSS primary and, as the
    # sensitivity arm, on the homology split; the CDS picks on the disjoint split
    # (the CDS side of the CDS-TSS pairs).
    intervals: dict = {}
    for task in ("family5", "genept"):
        cds = R.cells(cds_recs, "cds", task)
        srcs = {*R.NT_KMERS, "codon", "gc", *R.AA_KMERS, "esm2_150m", "esm2_650m",
                *R.NT_KMERS_LEN, *R.AA_KMERS_LEN, *(R.best_pool(cds, e, "cds") for e in R.ENCODERS)}
        for src in sorted(srcs):
            intervals[f"{CDS}/{task}/{src}"] = ci(cds_recs, "cds", task, src)
        dcds = R.cells(tss_recs, "cds", task)
        for src in sorted({"kmer", *(R.best_pool(dcds, e, "cds") for e in R.ENCODERS)}):
            intervals[f"{TSS}/{task}/{src}"] = ci(tss_recs, "cds", task, src)
        for split, recs in ((TSS, tss_recs), (CDS, cds_recs)):
            tss = R.cells(recs, "tss", task)
            srcs = {"enformer_tss_4mer", "enformer_trunk_global", "enformer_trunk_center",
                    *(f"tss_{e}_tssanchored" for e in R.ENCODERS),
                    *(R.best_pool(tss, e, "tss") for e in R.ENCODERS)}
            for src in sorted(srcs):
                intervals[f"{split}/{task}/{src}"] = ci(recs, "tss", task, src)
    result["intervals"] = intervals

    # Exploratory paired tests (unadjusted).
    expl: dict = {}
    for task in ("family5", "genept"):
        h = _headline(cds_recs, task)
        pairs = [(f"{task}: aa_kmer > nt_kmer", task, h["aa_kmer"], h["nt_kmer"]),
                 (f"{task}: aa2 > kmer", task, "aa2", "kmer"),
                 (f"{task}: esm2_650m > esm2_150m", task, "esm2_650m", "esm2_150m"),
                 (f"{task}: esm2_650m > aa_kmer", task, "esm2_650m", h["aa_kmer"])]
        cds = R.cells(cds_recs, "cds", task)   # Rule 3: does encoder > composition survive length?
        pairs += [(f"{task}: encoder > nt_kmer+len", task, h["best_encoder"], R.best_nt_kmer_len(cds)),
                  (f"{task}: encoder > aa_kmer+len", task, h["best_encoder"], R.best_aa_len(cds))]
        if task == "genept":   # the regression analogues of T1-T3, and the AA ladder (R1)
            pairs += [(f"{task}: encoder > nt_kmer", task, h["best_encoder"], h["nt_kmer"]),
                      (f"{task}: encoder > aa_kmer", task, h["best_encoder"], h["aa_kmer"]),
                      (f"{task}: esm2_650m > encoder", task, "esm2_650m", h["best_encoder"]),
                      (f"{task}: aa3 > aa2", task, "aa3", "aa2"),
                      (f"{task}: aa2 > aa1", task, "aa2", "aa1")]
        expl.update({f"{CDS} {k}": v for k, v in exploratory_pairs(cds_recs, pairs, n).items()})
        for split, recs in ((TSS, tss_recs), (CDS, cds_recs)):
            tss = R.cells(recs, "tss", task)
            dcds = R.cells(recs, "cds", task)
            tss_pairs = [(f"{task}: kmer CDS > TSS 4-mer (control)", task, "kmer", "enformer_tss_4mer"),
                         (f"{task}: Enformer centre > whole", task, "enformer_trunk_center",
                          "enformer_trunk_global")]
            for e in R.ENCODERS:
                g = R.best_pool(tss, e, "tss")
                tss_pairs += [(f"{task}: {e} CDS > TSS", task, R.best_pool(dcds, e, "cds"), g),
                              (f"{task}: {e} anchored > whole-window", task, f"tss_{e}_tssanchored", g),
                              (f"{task}: {e} anchored > anchored-chunk composition", task,
                               f"tss_{e}_tssanchored",
                               R.pick(tss, [f"tss_{e}_chunk4mergc", f"tss_{e}_chunk6mer"]))]
            expl.update({f"{split} {k}": v for k, v in exploratory_pairs(recs, tss_pairs, n).items()})
    result["exploratory"] = expl
    # Rule 3 additions, exploratory: each interval cell's own chance level, GenePT
    # R^2 beyond the family, and T1-T3 on the re-seeded CDS splits.
    result["chance"] = {k: stats.permutation_chance(rec, n_iters=n) for k, rec in sorted(interval_recs.items())}
    result["within_family"] = within_family(
        {k: r for k, r in interval_recs.items()
         if (r["split"], r["arm"], r["task"]) == (CDS, "cds", "genept")}, n)
    result["seed_replication"] = seed_replication(seeds, n)
    result["sensitivity"] = sensitivity(cds_recs, tss_recs, n)
    result["pooling_3x"] = pooling_3x(cds_recs, n)

    # Null bands (G13).
    bands = {}
    for split in (CDS, TSS):
        null = nulls[split]
        by_cell: dict[tuple, list[dict]] = {}
        for r in null.values():
            by_cell.setdefault((r["task"], r["feature_source"]), []).append(r)
        for (task, src), rs in sorted(by_cell.items()):
            bands[f"{split}/{task}/{src}"] = stats.null_band(rs, NULL_SHUFFLES)
    result["null_bands"] = bands

    stats.write_json(args.out, result)
    print(f"wrote {args.out}")
    for k, v in result["confirmatory"].items():
        lo, hi = v["delta_ci95"]
        print(f"  {k}: delta {v['delta_point']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"p={v['p_one_sided']:.4f}  Holm={v['p_holm']:.4f}")


if __name__ == "__main__":
    main()
