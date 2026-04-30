"""Compute Cohen's kappa for the headline classification cells.

For 5-way: kappa is computed directly from the saved confusion matrix
(no probe re-run needed).

For binary tasks: refits the logistic probe at the recorded best C and
computes kappa from y_true / y_pred via sklearn.

Output: a markdown table with macro-F1, accuracy, anti-baseline F1, and
Cohen's kappa per cell. Saves to data/kappa_summary.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score

from binary_tasks import load_binary_split
from kmer_baseline import featurize_cds
from data_loader.sequence_fetcher import fetch_cds
from length_baseline import cds_length_features
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SEQUENCES_DIR = DATA / "sequences"
OUT = DATA / "kappa_summary.md"

DATASET_PATHS = {
    "dnabert2": DATA / "dataset.parquet",
    "nt_v2": DATA / "dataset_nt_v2.parquet",
}
for _enc in ("dnabert2", "nt_v2"):
    for _v in ("meanmean", "maxmean", "clsmean", "meanD", "meanG"):
        DATASET_PATHS[f"{_enc}_{_v}"] = DATA / f"dataset_{_enc}_{_v}.parquet"
del _enc, _v
META_PARQUET = DATASET_PATHS["dnabert2"]


def kappa_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    po = np.trace(cm) / n
    p_true = cm.sum(axis=1) / n
    p_pred = cm.sum(axis=0) / n
    pe = float((p_true * p_pred).sum())
    return float((po - pe) / (1.0 - pe))


def _kmer_features(meta):
    out = np.zeros((len(meta), 256), dtype=np.float32)
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, SEQUENCES_DIR)
        if not seq:
            raise RuntimeError(f"missing CDS for {eid}")
        out[i] = featurize_cds(seq)
    return out


def _length_features(meta):
    return cds_length_features(meta)


def _load(dataset, task, name):
    if task == "family5":
        if dataset in DATASET_PATHS:
            X, _, meta = load_split(name, dataset_path=DATASET_PATHS[dataset])
        else:
            _, _, meta = load_split(name, dataset_path=META_PARQUET)
            X = _kmer_features(meta) if dataset == "kmer" else _length_features(meta)
        y = meta["family"].to_numpy()
        return X, y
    if dataset in DATASET_PATHS:
        X, y, _ = load_binary_split(task, name, dataset_path=DATASET_PATHS[dataset])
    else:
        _, y, meta = load_binary_split(task, name, dataset_path=META_PARQUET)
        X = _kmer_features(meta) if dataset == "kmer" else _length_features(meta)
    return X, y


def kappa_for_binary(dataset: str, task: str, C: float, shuffle: bool = False) -> dict:
    X_tr, y_tr = _load(dataset, task, "train")
    X_val, y_val = _load(dataset, task, "val")
    X_te, y_te = _load(dataset, task, "test")
    if shuffle:
        rng = np.random.default_rng(42)
        y_tr = rng.permutation(y_tr)
        y_val = rng.permutation(y_val)
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr, y_val])
    probe = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X_fit, y_fit)
    y_pred = probe.predict(X_te)
    return {
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_te, y_pred)),
    }


# Cells to include in the headline kappa table. Best C is pulled from
# the latest metrics.json entry per (encoder, task) so kappa numbers
# match the recorded macro-F1 exactly.
HEADLINE_CELLS = [
    ("nt_v2_meanD",       "family5"),
    ("nt_v2",             "family5"),
    ("nt_v2_meanmean",    "family5"),
    ("dnabert2_meanD",    "family5"),
    ("dnabert2_meanmean", "family5"),
    ("dnabert2",          "family5"),
    ("kmer",              "family5"),
    ("nt_v2_clsmean",     "family5"),
    ("nt_v2_maxmean",     "family5"),
    ("dnabert2_meanmean", "tf_vs_gpcr"),
    ("nt_v2_meanD",       "tf_vs_gpcr"),
    ("nt_v2",             "tf_vs_gpcr"),
    ("dnabert2",          "tf_vs_gpcr"),
    ("kmer",              "tf_vs_gpcr"),
    ("dnabert2_meanmean", "tf_vs_kinase"),
    ("dnabert2_meanD",    "tf_vs_kinase"),
    ("nt_v2",             "tf_vs_kinase"),
    ("kmer",              "tf_vs_kinase"),
]


def _recorded_best_C(metrics: list[dict], encoder: str, task: str,
                      shuffled: bool = False) -> float:
    """Latest entry per (encoder, task, shuffled_labels) — same dedupe
    logic as the inspection scripts."""
    best = None
    for r in metrics:
        if r.get("model") != "logistic_probe":
            continue
        if r["encoder"] != encoder or r["task"] != task:
            continue
        if bool(r.get("shuffled_labels", False)) != shuffled:
            continue
        if best is None or r["timestamp"] > best["timestamp"]:
            best = r
    if best is None:
        raise KeyError(f"no metrics entry for ({encoder}, {task}, shuffled={shuffled})")
    return float(best["C"])


def main():
    metrics = json.loads((DATA / "metrics.json").read_text())

    rows = []
    for dataset, task in HEADLINE_CELLS:
        C = _recorded_best_C(metrics, dataset, task, shuffled=False)
        if task == "family5":
            cm_path = DATA / f"confusion_5way_{dataset}.json"
            if cm_path.exists():
                cm = np.array(json.loads(cm_path.read_text())["matrix"])
                kappa = kappa_from_confusion(cm)
                rows.append({"task": task, "dataset": dataset, "C": C,
                             "kappa": kappa, "via": "confusion_matrix"})
                print(f"  {task:<14s} {dataset:<22s} C={C:<8g} kappa={kappa:.4f}  (cm)")
                continue
        res = kappa_for_binary(dataset, task, C)
        rows.append({"task": task, "dataset": dataset, "C": C,
                     "kappa": res["kappa"], "via": "refit"})
        print(f"  {task:<14s} {dataset:<22s} C={C:<8g} kappa={res['kappa']:.4f}  (refit)")

    print("\n--- anti-baselines (shuffled labels) ---")
    anti_rows = []
    for task in ("family5", "tf_vs_gpcr", "tf_vs_kinase"):
        C = _recorded_best_C(metrics, "shuffled", task, shuffled=True)
        res = kappa_for_binary("nt_v2", task, C, shuffle=True)
        anti_rows.append({"task": task, "dataset": "shuffled", "C": C,
                          "kappa": res["kappa"], "macro_f1": res["macro_f1"]})
        print(f"  {task:<14s} shuffled                 C={C:<8g} kappa={res['kappa']:+.4f}  macro_f1={res['macro_f1']:.4f}")

    # Pull macro-F1 from metrics.json for the main cells (latest dedup)
    runs = [r for r in json.loads((DATA / "metrics.json").read_text())
            if r.get("model") == "logistic_probe" and not r.get("shuffled_labels")]
    seen = {}
    for r in runs:
        seen[(r["encoder"], r["task"])] = r
    by_cell = seen

    OUT.write_text("")  # truncate
    with OUT.open("w") as f:
        f.write("# Cohen's kappa for headline classification cells\n\n")
        f.write(
            "Cohen's kappa = (p_observed − p_expected) / (1 − p_expected). "
            "Chance-corrected: 0 = chance, 1 = perfect. Computed from the saved "
            "5-way confusion matrices (`data/confusion_5way_*.json`); for the "
            "binary tasks we refit the probe at the recorded best C and compute "
            "kappa via `sklearn.metrics.cohen_kappa_score`.\n\n"
        )
        for task in ("family5", "tf_vs_gpcr", "tf_vs_kinase"):
            f.write(f"## {task}\n\n")
            f.write("| Dataset | C | macro-F1 | Cohen's κ |\n|---|---:|---:|---:|\n")
            sub = [r for r in rows if r["task"] == task]
            sub_sorted = sorted(sub, key=lambda r: -r["kappa"])
            for r in sub_sorted:
                m_f1 = by_cell.get((r["dataset"], task), {}).get("test_macro_f1", float("nan"))
                f.write(f"| `{r['dataset']}` | {r['C']:g} | {m_f1:.4f} | **{r['kappa']:.4f}** |\n")
            anti = next((a for a in anti_rows if a["task"] == task), None)
            if anti:
                f.write(f"| `shuffled-label` (anti-baseline) | {anti['C']:g} | "
                        f"{anti['macro_f1']:.4f} | {anti['kappa']:+.4f} |\n")
            f.write("\n")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
