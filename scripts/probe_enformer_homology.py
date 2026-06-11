#!/usr/bin/env python3
"""Re-probe the (split-independent) Enformer features on the homology split.

The Enformer comparator was only probed on the random split in the original
run; its per-gene features are cached and split-independent, so this re-fits
the family5 logistic probe and the Ridge-to-GenePT probe on the homology
train+val and evaluates on the homology test set, matching the protocol used
for the other homology comparators. Writes data/metrics_enformer_homology.json.

    uv run scripts/probe_enformer_homology.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent.parent / "data"
SUMMARIES = {"enformer_trunk_global": "dataset_enformer_trunk_global.parquet",
             "enformer_trunk_center": "dataset_enformer_trunk_center.parquet"}
GRID = [0.01, 0.1, 1, 10, 100, 1000]
FAMS = ["gpcr", "immune", "ion", "kinase", "tf"]


def _masks(ids):
    spl = json.loads((DATA / "splits.json").read_text())
    sset = {k: set(spl[k]) for k in ("train", "val", "test")}
    return {k: np.array([i in s for i in ids]) for k, s in sset.items()}


def probe(parquet):
    df = pd.read_parquet(DATA / parquet)
    ids = df["ensembl_id"].astype(str).values
    X = np.stack(df["x"].values)
    Y = np.stack(df["y"].values)
    fam = df["family"].map({f: i for i, f in enumerate(FAMS)}).values
    m = _masks(ids)
    trv = m["train"] | m["val"]

    bestC = max(GRID, key=lambda C: f1_score(
        fam[m["val"]], LogisticRegression(C=C, max_iter=3000).fit(X[m["train"]], fam[m["train"]]).predict(X[m["val"]]),
        average="macro"))
    clf = LogisticRegression(C=bestC, max_iter=3000).fit(X[trv], fam[trv])
    p = clf.predict(X[m["test"]])
    f1 = float(f1_score(fam[m["test"]], p, average="macro"))
    kap = float(cohen_kappa_score(fam[m["test"]], p))

    besta = max(GRID, key=lambda a: r2_score(
        Y[m["val"]], Ridge(alpha=a).fit(X[m["train"]], Y[m["train"]]).predict(X[m["val"]]),
        multioutput="uniform_average"))
    r2 = float(r2_score(Y[m["test"]], Ridge(alpha=besta).fit(X[trv], Y[trv]).predict(X[m["test"]]),
                        multioutput="uniform_average"))
    return f1, kap, bestC, r2, besta


def main():
    out = []
    for src, parquet in SUMMARIES.items():
        f1, kap, C, r2, a = probe(parquet)
        out.append({"model": "logistic_probe", "feature_source": src, "task": "family5",
                    "shuffled_labels": False, "C": C, "test_macro_f1": f1, "test_kappa": kap})
        out.append({"model": "linear_probe", "dataset": f"dataset_{src}.parquet",
                    "select_by": "r2", "alpha": a, "test_r2_macro": r2})
        print(f"{src:24s} cls F1 {f1:.3f} / k {kap:.3f} | reg R2 {r2:.3f}")
    (DATA / "metrics_enformer_homology.json").write_text(json.dumps(out, indent=2))
    print("wrote data/metrics_enformer_homology.json")


if __name__ == "__main__":
    main()
