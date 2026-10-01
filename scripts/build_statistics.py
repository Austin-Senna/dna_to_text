"""Every interval, paired test and null band the paper reports (Phase 1D, D4).

Reads the camera-ready records (``data/v2``, via ``linear_trainer.records``),
picks each headline cell on validation scores only, and runs the statistics in
``linear_trainer.stats`` on their stored predictions. Writes
``data/v2/statistics.json``.

Confirmatory family (Holm-adjusted together; decided Sept 29), family5 macro-F1,
each side the validation-selected cell, one-sided for A > B:

  T1  best DNA encoder (CDS)  vs  nucleotide k-mer (k in {4, 6} chosen on val)
  T2  best DNA encoder (CDS)  vs  amino-acid k-mer (k in {1, 2, 3} chosen on val)
  T3  ESM-2 650M              vs  best DNA encoder (CDS)
  T4  best encoder on CDS     vs  the same encoder on TSS (both on the disjoint split)

T1-T3 run on the CDS primary split (``splits.json``); T4 on the TSS primary
(``splits_tss_disjoint.json``), where the CDS cells also run, so both sides
score the same test genes. Everything else is exploratory and unadjusted.

Run: uv run scripts/build_statistics.py [--n-iters 1000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from linear_trainer import records as R
from linear_trainer import stats

CDS = "splits.json"
TSS = "splits_tss_disjoint.json"
NULL_SHUFFLES = 200
OUT = R.V2 / "statistics.json"


def _headline(recs: dict, task: str) -> dict[str, str]:
    cds = R.cells(recs, "cds", task)
    return {"best_encoder": R.best_encoder(cds, "cds"), "nt_kmer": R.best_nt_kmer(cds),
            "aa_kmer": R.best_aa(cds), "esm2": "esm2_650m",
            **{f"best_{e}": R.best_pool(cds, e, "cds") for e in R.ENCODERS}}


def confirmatory(cds_recs: dict, tss_recs: dict, n_iters: int) -> dict:
    cds = R.cells(cds_recs, "cds", "family5")
    h = _headline(cds_recs, "family5")
    enc = h["best_encoder"]
    tests = {
        "T1 encoder > nucleotide k-mer": (cds[enc], cds[h["nt_kmer"]]),
        "T2 encoder > amino-acid k-mer": (cds[enc], cds[h["aa_kmer"]]),
        "T3 ESM-2 650M > encoder": (cds["esm2_650m"], cds[enc]),
    }
    # T4 on the disjoint split: the encoder picked on its own CDS validation there,
    # against that encoder's own TSS pick.
    dcds = R.cells(tss_recs, "cds", "family5")
    dtss = R.cells(tss_recs, "tss", "family5")
    enc_d = R.best_encoder(dcds, "cds")     # may differ from T1-T3's; the output names both
    tests["T4 CDS > TSS (same encoder)"] = (dcds[enc_d], dtss[R.best_pool(dtss, R.encoder_of(enc_d), "tss")])

    out = {k: stats.paired_bootstrap(a, b, n_iters=n_iters) for k, (a, b) in tests.items()}
    # T1-T3 must use one encoder cell; T4 names its own (picked on the disjoint split).
    adj = stats.holm({k: v["p_one_sided"] for k, v in out.items()})
    for k in out:
        out[k]["p_holm"] = adj[k]
    return out


def exploratory_pairs(recs: dict, arm_pairs: list[tuple[str, str, str, str]], n_iters: int) -> dict:
    out = {}
    for label, task, a, b in arm_pairs:
        cells_ = {**R.cells(recs, "cds", task), **R.cells(recs, "tss", task)}
        missing = [x for x in (a, b) if x not in cells_]
        if missing:
            raise R.MissingRecord(f"{label}: no record for {missing}")
        out[label] = stats.paired_bootstrap(cells_[a], cells_[b], n_iters=n_iters)
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
    stamp = R.stamp_of(cds_recs, tss_recs, *nulls.values())          # G7 across every file used
    result: dict = {"stamp": stamp, "n_iters": n, "seed": stats.SEED,
                    "confirmatory": confirmatory(cds_recs, tss_recs, n)}

    def ci(recs, arm, task, src):
        return stats.cluster_bootstrap(R.cells(recs, arm, task)[src], n_iters=n)

    # Intervals: every CDS comparator and encoder pick on the CDS primary; every
    # TSS pick, Enformer readout and E5 cell on the TSS primary and, as the
    # sensitivity arm, on the homology split; the CDS picks on the disjoint split
    # (the CDS side of the CDS-TSS pairs).
    intervals: dict = {}
    for task in ("family5", "genept"):
        cds = R.cells(cds_recs, "cds", task)
        srcs = {*R.NT_KMERS, "codon", "gc", *R.AA_KMERS, "esm2_150m", "esm2_650m",
                *(R.best_pool(cds, e, "cds") for e in R.ENCODERS)}
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
