"""Bootstrap test-set 95% CIs for the headline classification and regression cells.

For each cell named in HEADLINE_CLS / HEADLINE_REG below:
  1. Refit the probe at the recorded best hyperparameter on train+val
     (matching the original protocol exactly).
  2. Predict on the held-out test split.
  3. Bootstrap-resample the test set 1,000 iterations (stratified by family
     for classification; iid by gene for regression) and recompute the
     metric on each resample.
  4. Report 95% percentile CIs.

Also computes per-class F1 from the full test predictions for each
classification cell (no bootstrap needed; just the standard per-class F1).

Output: data/bootstrap_metrics.json (and a stdout summary table).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score

from linear_trainer import fit_logistic
from kmer_baseline import featurize_cds
from data_loader.sequence_fetcher import fetch_cds
from splits import load_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SEQUENCES_DIR = DATA / "sequences"
OUT = DATA / "bootstrap_metrics.json"

# All encoder dataset parquets we might bootstrap. Keep in sync with compute_kappa.py.
DATASET_PATHS: dict[str, Path] = {}
for _enc in ("dnabert2", "nt_v2", "gena_lm", "hyena_dna"):
    _base = DATA / f"dataset_{_enc}.parquet"
    if _base.exists():
        DATASET_PATHS[_enc] = _base
    for _v in ("meanmean", "specialmean", "maxmean", "clsmean", "meanD", "meanG"):
        _p = DATA / f"dataset_{_enc}_{_v}.parquet"
        if _p.exists():
            DATASET_PATHS[f"{_enc}_{_v}"] = _p
    # TSS counterparts (only present for encoders that have run the TSS pipeline)
    _tss_base = DATA / f"dataset_tss_{_enc}.parquet"
    if _tss_base.exists():
        DATASET_PATHS[f"tss_{_enc}"] = _tss_base
    for _v in ("meanmean", "specialmean", "maxmean", "clsmean", "meanD", "meanG"):
        _p = DATA / f"dataset_tss_{_enc}_{_v}.parquet"
        if _p.exists():
            DATASET_PATHS[f"tss_{_enc}_{_v}"] = _p
# TSS 4-mer baseline lives under enformer_tss_4mer because the prep pipeline
# computed it alongside the Enformer comparator.
_tss_kmer = DATA / "dataset_enformer_tss_4mer.parquet"
if _tss_kmer.exists():
    DATASET_PATHS["enformer_tss_4mer"] = _tss_kmer
del _enc, _v, _p, _base, _tss_base, _tss_kmer

# Use any tracked encoder parquet for metadata-only loads (kmer baseline). All
# encoder parquets share the same {ensembl_id, family, gene_symbol, summary,
# genept_vec} columns; they differ only in the embedding column.
META_PARQUET = DATA / "dataset_dnabert2_meanmean.parquet"

# Headline cells to bootstrap. Each tuple is (cell_name, recorded_hyperparam,
# is_shuffled). Hyperparameter values come from the latest entry in
# data/metrics.json that matches (encoder, task, shuffled flag).
HEADLINE_CLS = [
    # (cell_name, dataset_for_X_loading, recorded_C, shuffled_labels)
    ("kmer",                "kmer",                  1000.0,  False),
    ("dnabert2_meanD",      "dnabert2_meanD",        10.0,    False),
    ("nt_v2_meanD",         "nt_v2_meanD",           1.0,     False),
    ("gena_lm_clsmean",     "gena_lm_clsmean",       1.0,     False),
    ("hyena_dna_meanG",     "hyena_dna_meanG",       10.0,    False),
    ("shuffled",            "nt_v2_meanD",           100.0,   True),
]

HEADLINE_REG = [
    # (cell_name, dataset_for_X_loading, recorded_alpha, shuffled_y)
    ("kmer",                "kmer",                  0.01,    False),
    ("dnabert2_meanG",      "dnabert2_meanG",        10.0,    False),
    ("nt_v2_meanmean",      "nt_v2_meanmean",        10.0,    False),
    ("gena_lm_meanmean",    "gena_lm_meanmean",      100.0,   False),
    ("hyena_dna_specialmean","hyena_dna_specialmean", 1.0,    False),
    ("shuffled_y",          "nt_v2_meanmean",        1000.0,  True),
]

# TSS-context headline cells (added during the revision/tss-and-gene-scope work).
# Hyperparameters are the recorded best from each encoder's TSS probe sweep.
HEADLINE_CLS_TSS = [
    # (cell_name,                  dataset_for_X_loading,       recorded_C, shuffled_labels)
    ("tss_4mer",                   "enformer_tss_4mer",         1000.0,     False),
    ("tss_nt_v2_meanmean",         "tss_nt_v2_meanmean",        100.0,      False),
    ("tss_hyena_dna_meanmean",     "tss_hyena_dna_meanmean",    1000.0,     False),
    ("tss_dnabert2_maxmean",       "tss_dnabert2_maxmean",      100.0,      False),
    # GENA-LM TSS row is filled in once its probe sweep finishes. Best pool
    # and C come from data/metrics.json under model=logistic_probe, task=family5,
    # encoder/feature_source starts with tss_gena_lm_. Format:
    # ("tss_gena_lm_<best>",       "tss_gena_lm_<best>",        <C>,        False),
]

HEADLINE_REG_TSS = [
    # (cell_name,                  dataset_for_X_loading,       recorded_alpha, shuffled_y)
    ("tss_4mer",                   "enformer_tss_4mer",         0.01,           False),
    ("tss_nt_v2_meanmean",         "tss_nt_v2_meanmean",        0.1,            False),
    ("tss_hyena_dna_meanmean",     "tss_hyena_dna_meanmean",    0.1,            False),
    ("tss_dnabert2_meanmean",      "tss_dnabert2_meanmean",     0.1,            False),
    # GENA-LM TSS regression row is filled in after its ridge sweep. Find the
    # max-R² entry in data/metrics.json (model=linear_probe, dataset starts
    # with dataset_tss_gena_lm_). Format:
    # ("tss_gena_lm_<best>",       "tss_gena_lm_<best>",        <alpha>,        False),
]


def _kmer_features(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 256), dtype=np.float32)
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, SEQUENCES_DIR)
        if not seq:
            raise RuntimeError(f"missing CDS for {eid}")
        out[i] = featurize_cds(seq)
    return out


def _load(dataset: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_family, Y_genept) for the given dataset/split."""
    if dataset in DATASET_PATHS:
        X, Y_genept, meta = load_split(name, dataset_path=DATASET_PATHS[dataset])
    elif dataset == "kmer":
        _, Y_genept, meta = load_split(name, dataset_path=META_PARQUET)
        X = _kmer_features(meta)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    y_family = meta["family"].to_numpy()
    return X, y_family, Y_genept


def bootstrap_classification(dataset: str, C: float, shuffled: bool,
                              n_iters: int = 1000, seed: int = 42) -> dict:
    X_tr, y_tr, _ = _load(dataset, "train")
    X_val, y_val, _ = _load(dataset, "val")
    X_te, y_te, _ = _load(dataset, "test")
    if shuffled:
        rs = np.random.default_rng(seed)
        y_tr = rs.permutation(y_tr)
        y_val = rs.permutation(y_val)
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr, y_val])
    probe = fit_logistic(X_fit, y_fit, C)
    y_pred = probe.predict(X_te)

    # Stratified bootstrap by true class
    classes = sorted(np.unique(y_te).tolist())
    rng = np.random.default_rng(seed)
    f1s, kappas = [], []
    for _ in range(n_iters):
        parts = []
        for c in classes:
            ci = np.where(y_te == c)[0]
            parts.append(rng.choice(ci, size=len(ci), replace=True))
        idx = np.concatenate(parts)
        f1s.append(f1_score(y_te[idx], y_pred[idx], average="macro"))
        kappas.append(cohen_kappa_score(y_te[idx], y_pred[idx]))

    per_class = f1_score(y_te, y_pred, average=None, labels=classes)
    return {
        "n_test": int(len(y_te)),
        "macro_f1_point": float(f1_score(y_te, y_pred, average="macro")),
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)),
                          float(np.percentile(f1s, 97.5))],
        "kappa_point": float(cohen_kappa_score(y_te, y_pred)),
        "kappa_ci95": [float(np.percentile(kappas, 2.5)),
                       float(np.percentile(kappas, 97.5))],
        "per_class_f1": {c: float(per_class[i]) for i, c in enumerate(classes)},
        "n_iters": n_iters,
    }


def bootstrap_regression(dataset: str, alpha: float, shuffled: bool,
                          n_iters: int = 1000, seed: int = 42) -> dict:
    X_tr, _, Y_tr = _load(dataset, "train")
    X_val, _, Y_val = _load(dataset, "val")
    X_te, _, Y_te = _load(dataset, "test")
    if shuffled:
        rs = np.random.default_rng(seed)
        perm_tr = rs.permutation(len(Y_tr))
        perm_val = rs.permutation(len(Y_val))
        Y_tr = Y_tr[perm_tr]
        Y_val = Y_val[perm_val]
    X_fit = np.vstack([X_tr, X_val])
    Y_fit = np.vstack([Y_tr, Y_val])
    probe = Ridge(alpha=alpha).fit(X_fit, Y_fit)
    Y_pred = probe.predict(X_te)

    rng = np.random.default_rng(seed)
    n = len(X_te)
    r2s = []
    for _ in range(n_iters):
        idx = rng.choice(n, size=n, replace=True)
        Y_te_b = Y_te[idx]
        Y_pred_b = Y_pred[idx]
        ss_res = np.sum((Y_te_b - Y_pred_b) ** 2, axis=0)
        ss_tot = np.sum((Y_te_b - Y_te_b.mean(axis=0)) ** 2, axis=0)
        r2_per = 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
        r2s.append(float(np.mean(r2_per)))

    return {
        "n_test": int(len(Y_te)),
        "r2_macro_point": float(np.mean(
            r2_score(Y_te, Y_pred, multioutput="raw_values"))),
        "r2_macro_ci95": [float(np.percentile(r2s, 2.5)),
                          float(np.percentile(r2s, 97.5))],
        "n_iters": n_iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    results = {"classification": {}, "regression": {},
               "n_iters": args.n_iters, "seed": args.seed}

    print("=== Classification bootstrap CIs (family5) ===")
    for name, dataset, C, shuf in HEADLINE_CLS + HEADLINE_CLS_TSS:
        t0 = time.time()
        res = bootstrap_classification(dataset, C, shuf,
                                        n_iters=args.n_iters, seed=args.seed)
        results["classification"][name] = res
        f1_lo, f1_hi = res["macro_f1_ci95"]
        k_lo, k_hi = res["kappa_ci95"]
        per = res["per_class_f1"]
        print(f"  {name:<26s} F1={res['macro_f1_point']:.4f} [{f1_lo:.3f}-{f1_hi:.3f}]  "
              f"kappa={res['kappa_point']:.4f} [{k_lo:.3f}-{k_hi:.3f}]  "
              f"({time.time()-t0:.1f}s)")
        per_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(per.items()))
        print(f"    per-class F1: {per_str}")

    print("\n=== Regression bootstrap CIs (Ridge -> GenePT) ===")
    for name, dataset, alpha, shuf in HEADLINE_REG + HEADLINE_REG_TSS:
        t0 = time.time()
        res = bootstrap_regression(dataset, alpha, shuf,
                                    n_iters=args.n_iters, seed=args.seed)
        results["regression"][name] = res
        r2_lo, r2_hi = res["r2_macro_ci95"]
        print(f"  {name:<26s} R2={res['r2_macro_point']:.4f} [{r2_lo:.3f}-{r2_hi:.3f}]  "
              f"({time.time()-t0:.1f}s)")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
