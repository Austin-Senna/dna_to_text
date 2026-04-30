"""Train a logistic regression classification probe.

One script for the entire 15-cell run matrix. Flags pick which feature
source X comes from (--dataset) and which classification task (--task);
--shuffle-labels turns the run into the anti-baseline.

Outputs:
  - one entry appended to data/metrics.json per run
  - for the 5-way task, a confusion matrix saved as a JSON file
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from binary_tasks import load_binary_split, BINARY_TASKS
from kmer_baseline import featurize_cds
from data_loader.sequence_fetcher import fetch_cds
from length_baseline import cds_length_features
from linear_trainer import fit_logistic, sweep_C
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SEQUENCES_DIR = DATA / "sequences"

DEFAULT_CS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

DATASET_PATHS = {
    "dnabert2": DATA / "dataset.parquet",
    "nt_v2":    DATA / "dataset_nt_v2.parquet",
}
# Phase 4b pooling variants (one parquet per encoder x variant).
for _enc in ("dnabert2", "nt_v2"):
    for _v in ("meanmean", "maxmean", "clsmean", "meanD", "meanG"):
        DATASET_PATHS[f"{_enc}_{_v}"] = DATA / f"dataset_{_enc}_{_v}.parquet"
del _enc, _v

META_PARQUET = DATASET_PATHS["dnabert2"]


def _kmer_features_for_meta(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 256), dtype=np.float32)
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, SEQUENCES_DIR)
        if not seq:
            raise RuntimeError(f"missing cached CDS for {eid}")
        out[i] = featurize_cds(seq)
    return out


def _length_features_for_meta(meta: pd.DataFrame) -> np.ndarray:
    return cds_length_features(meta)


def _load_split_for(
    dataset: str,
    task: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, meta) for the given (dataset, task, name) cell."""
    if task == "family5":
        if dataset in DATASET_PATHS:
            X, _, meta = load_split(name, dataset_path=DATASET_PATHS[dataset])
        else:
            _, _, meta = load_split(name, dataset_path=META_PARQUET)
            if dataset == "kmer":
                X = _kmer_features_for_meta(meta)
            elif dataset == "length":
                X = _length_features_for_meta(meta)
            else:
                raise ValueError(f"unknown dataset: {dataset!r}")
        y = meta["family"].to_numpy()
        return X, y, meta

    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")
    if dataset in DATASET_PATHS:
        X, y, meta = load_binary_split(task, name, dataset_path=DATASET_PATHS[dataset])
    else:
        _, y, meta = load_binary_split(task, name, dataset_path=META_PARQUET)
        if dataset == "kmer":
            X = _kmer_features_for_meta(meta)
        elif dataset == "length":
            X = _length_features_for_meta(meta)
        else:
            raise ValueError(f"unknown dataset: {dataset!r}")
    return X, y, meta


def _assert_splits_disjoint(meta_tr, meta_val, meta_te) -> None:
    s_tr, s_val, s_te = (
        set(meta_tr["ensembl_id"]),
        set(meta_val["ensembl_id"]),
        set(meta_te["ensembl_id"]),
    )
    assert s_tr.isdisjoint(s_val), "train/val overlap"
    assert s_tr.isdisjoint(s_te), "train/test overlap"
    assert s_val.isdisjoint(s_te), "val/test overlap"


def _assert_class_presence(y_tr, y_val, y_te, task: str) -> None:
    if task == "family5":
        for split_name, y in (("train", y_tr), ("val", y_val), ("test", y_te)):
            classes_present = set(np.unique(y).tolist())
            assert len(classes_present) == 5, (
                f"family5 split {split_name} has {len(classes_present)} classes, expected 5"
            )
    else:
        for split_name, y in (("train", y_tr), ("val", y_val), ("test", y_te)):
            bal = float(np.mean(y == 1))
            assert 0.45 <= bal <= 0.55, (
                f"binary split {split_name} class balance {bal:.3f} outside [0.45, 0.55]"
            )


def _eval_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    classes = sorted(set(y_true.tolist()))
    per_class = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            per_class[str(c)] = None
        else:
            per_class[str(c)] = float((y_pred[mask] == c).mean())
    return {
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": bal_acc,
        "test_accuracy": acc,
        "test_per_class_accuracy": per_class,
    }


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    classes = sorted(set(y_true.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"classes": classes, "matrix": cm}, indent=2))


def _append_metrics(path: Path, entry: dict) -> None:
    runs: list = []
    if path.exists():
        runs = json.loads(path.read_text())
        if not isinstance(runs, list):
            raise ValueError(f"{path} is not a JSON array")
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=sorted(set(DATASET_PATHS.keys()) | {"kmer", "length"}))
    ap.add_argument("--task", required=True,
                    choices=["family5", "tf_vs_gpcr", "tf_vs_kinase"])
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="anti-baseline: permute y in train+val before fit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--Cs", type=float, nargs="+", default=DEFAULT_CS)
    ap.add_argument("--metrics-out", default=str(DATA / "metrics.json"))
    args = ap.parse_args()

    print(f"=== loading splits: dataset={args.dataset} task={args.task} ===")
    X_tr,  y_tr,  meta_tr  = _load_split_for(args.dataset, args.task, "train")
    X_val, y_val, meta_val = _load_split_for(args.dataset, args.task, "val")
    X_te,  y_te,  meta_te  = _load_split_for(args.dataset, args.task, "test")
    print(f"  train={X_tr.shape} val={X_val.shape} test={X_te.shape}")

    _assert_splits_disjoint(meta_tr, meta_val, meta_te)
    _assert_class_presence(y_tr, y_val, y_te, args.task)

    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        y_tr_use = rng.permutation(y_tr)
        y_val_use = rng.permutation(y_val)
        print(f"  ANTI-BASELINE: shuffled labels in train+val (seed={args.seed})")
    else:
        y_tr_use, y_val_use = y_tr, y_val

    print("\n=== C sweep (macro_f1 on val) ===")
    best_C, sweep = sweep_C(X_tr, y_tr_use, X_val, y_val_use, args.Cs)
    for r in sweep:
        mark = " *" if r["C"] == best_C else ""
        print(f"  C={r['C']:>8.3g}  macro_f1={r['macro_f1']:.4f}{mark}")
    print(f"  best C = {best_C}")

    print("\n=== refit on train+val, evaluate on test ===")
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr_use, y_val_use])
    probe = fit_logistic(X_fit, y_fit, best_C)
    y_pred = probe.predict(X_te)
    metrics = _eval_test(y_te, y_pred)
    print(f"  test_macro_f1          = {metrics['test_macro_f1']:.4f}")
    print(f"  test_balanced_accuracy = {metrics['test_balanced_accuracy']:.4f}")
    print(f"  test_accuracy          = {metrics['test_accuracy']:.4f}")
    print(f"  per-class:               {metrics['test_per_class_accuracy']}")

    if args.task == "family5" and not args.shuffle_labels and args.dataset in DATASET_PATHS:
        cm_path = DATA / f"confusion_5way_{args.dataset}.json"
        _save_confusion_matrix(y_te, y_pred, cm_path)
        print(f"  wrote confusion matrix → {cm_path}")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    encoder_label = "shuffled" if args.shuffle_labels else args.dataset
    entry = {
        "run_id": f"logistic_{encoder_label}_{args.task}_{ts}",
        "timestamp": ts,
        "model": "logistic_probe",
        "encoder": encoder_label,
        "feature_source": args.dataset,
        "task": args.task,
        "shuffled_labels": args.shuffle_labels,
        "C": best_C,
        "C_sweep": sweep,
        **metrics,
    }
    _append_metrics(Path(args.metrics_out), entry)
    print(f"  appended metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
