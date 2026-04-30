# Classification Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot Phase 4 of the DNA-to-text project from regression (predict 1536-d GenePT y) to classification (predict family directly), reusing the existing cached embeddings. Run a 15-cell matrix (3 tasks × 5 feature sources) of logistic regression probes and baselines, then update docs.

**Architecture:** New `src/binary_tasks/` package builds two deterministic balanced binary subsets (tf-vs-gpcr, tf-vs-kinase) and exposes a loader. New `src/length_baseline/` package gives a one-feature length baseline. New `src/linear_trainer/logistic_probe.py` module wraps `sklearn.linear_model.LogisticRegression` with a C-sweep mirroring the Ridge probe's discipline. A single `scripts/train_logistic_probe.py` script handles every cell of the matrix via `--dataset {dnabert2,nt_v2,kmer,length}`, `--task {family5,tf_vs_gpcr,tf_vs_kinase}`, and `--shuffle-labels` flags. Results land in the existing `data/metrics.json` (additive schema).

**Tech Stack:** Python 3.11+, scikit-learn (`LogisticRegression`, `f1_score`, `balanced_accuracy_score`, `confusion_matrix`), numpy, pandas, pyarrow. Run via `uv run python scripts/...`. No new pip deps.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-04-29-classification-pivot-design.md`
- Phase 3 results: `findings.md`
- Existing patterns: `src/linear_trainer/probe.py`, `src/splits/loader.py`, `src/kmer_baseline/featurizer.py`

**Project test conventions:** This repo has no `pytest` setup and no `tests/` directory. Existing modules use embedded `assert` statements inside scripts as the sanity gate, plus end-to-end smoke runs that verify printed outputs. This plan follows the same pattern: each module gets a small `__main__` smoke check and each script has assertion gates that fail loud on bad data. **Do not introduce pytest** — it would diverge from the established style.

---

## File structure

**New files:**
```
src/binary_tasks/
  __init__.py                              # re-exports
  make_subsets.py                          # build deterministic downsampled binary subsets
  loader.py                                # load_binary_split(task, name, dataset_path)
src/length_baseline/
  __init__.py                              # re-exports
  featurizer.py                            # cds_length_features(meta) -> (n, 1)
src/linear_trainer/
  logistic_probe.py                        # C-sweep + fit + LogisticProbe dataclass
scripts/
  make_binary_subsets.py                   # CLI: builds the two JSON artefacts
  train_logistic_probe.py                  # CLI: handles all 15 matrix cells
data/
  binary_tf_gpcr.json                      # frozen subset + binary split (591/591)
  binary_tf_kinase.json                    # frozen subset + binary split (558/558)
  confusion_5way_dnabert2.json             # 5-way confusion matrix per encoder
  confusion_5way_nt_v2.json
```

**Modified files:**
```
src/linear_trainer/__init__.py             # add logistic_probe re-exports
findings.md                                 # Phase 4 results section
next_steps.md                               # mark Phase 4a done, refresh Phase 4b/4c
data/metrics.json                          # 15+ new entries appended
```

---

## Task 1: Add `binary_tasks` package — subset builder + loader

**Files:**
- Create: `src/binary_tasks/__init__.py`
- Create: `src/binary_tasks/make_subsets.py`
- Create: `src/binary_tasks/loader.py`
- Create: `scripts/make_binary_subsets.py`
- Produces: `data/binary_tf_gpcr.json`, `data/binary_tf_kinase.json`

### Step 1.1: Create the package init

- [ ] Create `src/binary_tasks/__init__.py`:

```python
from binary_tasks.make_subsets import (
    BinaryTask,
    BINARY_TASKS,
    build_binary_subset,
    write_binary_subset_json,
)
from binary_tasks.loader import load_binary_split

__all__ = [
    "BinaryTask",
    "BINARY_TASKS",
    "build_binary_subset",
    "write_binary_subset_json",
    "load_binary_split",
]
```

### Step 1.2: Implement `make_subsets.py`

- [ ] Create `src/binary_tasks/make_subsets.py` with the deterministic downsampler + frozen split:

```python
"""Build deterministic balanced binary subsets of the full corpus.

Each subset:
  - downsamples the larger class to match the smaller class size (seed=42)
  - then computes a stratified 70/15/15 split on the binary label (seed=42)
  - and freezes both the gene selection AND the split into one JSON file.

Every probe + baseline + anti-baseline run reads the same JSON, so the
comparison across runs is on the same genes in the same splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "dataset.parquet"

SEED = 42

BinaryTask = Literal["tf_vs_gpcr", "tf_vs_kinase"]

# task -> (positive_class_label, negative_class_label).
# We pick the smaller class as "positive" (label 1) by convention so the
# downsample target is unambiguous: downsample tf to len(other_class).
BINARY_TASKS: dict[str, tuple[str, str]] = {
    "tf_vs_gpcr": ("gpcr", "tf"),
    "tf_vs_kinase": ("kinase", "tf"),
}


def build_binary_subset(
    df: pd.DataFrame,
    task: BinaryTask,
    seed: int = SEED,
) -> dict:
    """Downsample the larger family to match the smaller, then make a stratified split.

    Returns a dict ready for json.dump (no numpy types).
    """
    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")
    pos_fam, neg_fam = BINARY_TASKS[task]

    # Sort by ensembl_id so the .iloc-based sampling is deterministic regardless
    # of the parquet row order (dataset.parquet and dataset_nt_v2.parquet may
    # differ in row order).
    pos = df[df["family"] == pos_fam].sort_values("ensembl_id").reset_index(drop=True)
    neg = df[df["family"] == neg_fam].sort_values("ensembl_id").reset_index(drop=True)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"empty family for task {task}: pos={len(pos)} neg={len(neg)}")

    n = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(len(pos), size=n, replace=False)
    neg_idx = rng.choice(len(neg), size=n, replace=False)
    pos_ids = sorted(pos.iloc[pos_idx]["ensembl_id"].tolist())
    neg_ids = sorted(neg.iloc[neg_idx]["ensembl_id"].tolist())

    all_ids = pos_ids + neg_ids
    labels = [1] * len(pos_ids) + [0] * len(neg_ids)

    train_ids, rest_ids, train_y, rest_y = train_test_split(
        all_ids, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.50, random_state=seed, stratify=rest_y
    )

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(all_ids)

    return {
        "task": task,
        "seed": seed,
        "positive_label": pos_fam,
        "negative_label": neg_fam,
        "n_per_class": n,
        "positive_ensembl_ids": pos_ids,
        "negative_ensembl_ids": neg_ids,
        "split": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }


def write_binary_subset_json(
    task: BinaryTask,
    out_path: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET,
    seed: int = SEED,
) -> dict:
    df = pd.read_parquet(dataset_path)
    payload = build_binary_subset(df, task, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
```

### Step 1.3: Implement `loader.py`

- [ ] Create `src/binary_tasks/loader.py`:

```python
"""Load X, y_binary, meta for a frozen binary subset built by make_subsets.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from binary_tasks.make_subsets import BinaryTask, BINARY_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

SplitName = Literal["train", "val", "test"]
META_COLUMNS = ["ensembl_id", "symbol", "family", "summary"]


def _subset_path(task: BinaryTask, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"binary_{task}.json"


def load_binary_split(
    task: BinaryTask,
    name: SplitName,
    dataset_path: str | Path,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y_binary, meta) for one split of a frozen binary subset.

    `dataset_path` selects which encoder's X is used (e.g. dataset.parquet for
    DNABERT-2, dataset_nt_v2.parquet for NT-v2). y_binary is int8: 1 if the
    gene is in `positive_ensembl_ids`, 0 otherwise.
    """
    if name not in ("train", "val", "test"):
        raise ValueError(f"unknown split name: {name!r}")
    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")

    payload = json.loads(_subset_path(task, data_dir).read_text())
    ids = payload["split"][name]
    pos_ids = set(payload["positive_ensembl_ids"])

    df = pd.read_parquet(dataset_path)
    df = df.set_index("ensembl_id").loc[ids].reset_index()

    X = np.stack(df["x"].to_numpy()).astype(np.float32)
    y = np.array([1 if eid in pos_ids else 0 for eid in df["ensembl_id"]], dtype=np.int8)
    meta = df[META_COLUMNS].reset_index(drop=True)
    return X, y, meta
```

### Step 1.4: Implement the CLI

- [ ] Create `scripts/make_binary_subsets.py`:

```python
"""Build the two frozen binary subsets and write them to data/."""
from __future__ import annotations

import argparse
from pathlib import Path

from binary_tasks import write_binary_subset_json, BINARY_TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for task in BINARY_TASKS:
        out = out_dir / f"binary_{task}.json"
        payload = write_binary_subset_json(task, out, dataset_path=args.dataset)
        n = payload["n_per_class"]
        sp = payload["split"]
        print(
            f"  {task}: n_per_class={n}  "
            f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}  "
            f"-> {out}"
        )


if __name__ == "__main__":
    main()
```

### Step 1.5: Run the CLI and verify the JSON files

- [ ] Run from repo root:

```bash
uv run python scripts/make_binary_subsets.py
```

Expected output (approximately):
```
  tf_vs_gpcr: n_per_class=591  train=826 val=178 test=178  -> data/binary_tf_gpcr.json
  tf_vs_kinase: n_per_class=558  train=781 val=167 test=168  -> data/binary_tf_kinase.json
```

- [ ] Verify both JSON files exist and parse:

```bash
uv run python -c "
import json
for t in ('tf_vs_gpcr', 'tf_vs_kinase'):
    p = json.load(open(f'data/binary_{t}.json'))
    sp = p['split']
    n = p['n_per_class']
    print(f'{t}: n_per_class={n} train+val+test={len(sp[\"train\"])+len(sp[\"val\"])+len(sp[\"test\"])}={2*n}')
    assert len(sp['train'])+len(sp['val'])+len(sp['test']) == 2*n
    assert set(sp['train']).isdisjoint(sp['val'])
    assert set(sp['train']).isdisjoint(sp['test'])
    assert set(sp['val']).isdisjoint(sp['test'])
print('OK')
"
```

Expected: prints two lines and "OK".

### Step 1.6: Smoke-test the loader on both encoders

- [ ] Run:

```bash
uv run python -c "
from binary_tasks import load_binary_split
for ds in ('data/dataset.parquet', 'data/dataset_nt_v2.parquet'):
    for task in ('tf_vs_gpcr', 'tf_vs_kinase'):
        for name in ('train', 'val', 'test'):
            X, y, meta = load_binary_split(task, name, dataset_path=ds)
            bal = y.mean()
            print(f'{ds.split(\"/\")[-1]} {task} {name}: X={X.shape} y={y.shape} pos_frac={bal:.3f}')
            assert X.ndim == 2 and X.shape[0] == y.shape[0] == len(meta)
            assert 0.45 <= bal <= 0.55, f'class imbalance: {bal}'
print('OK')
"
```

Expected: 12 lines and "OK". `pos_frac` in [0.45, 0.55] for every split.

### Step 1.7: Commit

- [ ] Commit:

```bash
git add src/binary_tasks scripts/make_binary_subsets.py data/binary_tf_gpcr.json data/binary_tf_kinase.json
git commit -m "binary_tasks: add subset builder, loader, and frozen tf-vs-{gpcr,kinase} JSON artefacts"
```

---

## Task 2: Add `length_baseline` package

**Files:**
- Create: `src/length_baseline/__init__.py`
- Create: `src/length_baseline/featurizer.py`

### Step 2.1: Create the package init

- [ ] Create `src/length_baseline/__init__.py`:

```python
from length_baseline.featurizer import cds_length_features, LENGTH_DIM

__all__ = ["cds_length_features", "LENGTH_DIM"]
```

### Step 2.2: Implement the featurizer

- [ ] Create `src/length_baseline/featurizer.py`:

```python
"""CDS-length-only baseline. Single feature: log(len(cds)+1).

Catches the "encoder is just a length proxy" failure mode. We use log-length
so the feature is on the same order of magnitude as a typical embedding
dimension and the logistic regression's regularisation is well-scaled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.sequence_fetcher import fetch_cds

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_DIR = REPO_ROOT / "data" / "sequences"

LENGTH_DIM = 1


def cds_length_features(
    meta: pd.DataFrame,
    sequences_dir: Path = SEQUENCES_DIR,
) -> np.ndarray:
    """Return (n, 1) float32 array of log1p(CDS length) for each row of meta."""
    if "ensembl_id" not in meta.columns:
        raise ValueError("meta must have an 'ensembl_id' column")

    out = np.zeros((len(meta), LENGTH_DIM), dtype=np.float32)
    missing: list[str] = []
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, sequences_dir)
        if not seq:
            missing.append(eid)
            continue
        out[i, 0] = np.log1p(len(seq))
    if missing:
        raise RuntimeError(
            f"missing cached CDS for {len(missing)} gene(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return out
```

### Step 2.3: Smoke-test the featurizer

- [ ] Run:

```bash
uv run python -c "
from splits import load_split
from length_baseline import cds_length_features
_, _, meta = load_split('test')
X = cds_length_features(meta)
print(f'X={X.shape} dtype={X.dtype} mean={X.mean():.3f} min={X.min():.3f} max={X.max():.3f}')
assert X.shape == (len(meta), 1)
assert X.min() > 0  # all CDS have length >= 1
print('OK')
"
```

Expected: prints stats and "OK". `mean` should be roughly 7-8 (log of a few thousand bases).

### Step 2.4: Commit

- [ ] Commit:

```bash
git add src/length_baseline
git commit -m "length_baseline: add log-length single-feature CDS baseline"
```

---

## Task 3: Add `logistic_probe` module + training script

**Files:**
- Create: `src/linear_trainer/logistic_probe.py`
- Modify: `src/linear_trainer/__init__.py`
- Create: `scripts/train_logistic_probe.py`

### Step 3.1: Implement `logistic_probe.py`

- [ ] Create `src/linear_trainer/logistic_probe.py`:

```python
"""Logistic regression classification probe with L2 + C-sweep.

Mirrors the discipline of probe.py (Ridge): sweep C on val, refit on
train+val, evaluate on test. Multi-class via multinomial softmax;
binary via the same path (sklearn handles the dispatch internally).

Headline metric for the C-sweep is macro-F1 (not accuracy) so the sweep
optimises for the same metric we report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


@dataclass
class LogisticProbe:
    W: np.ndarray         # (d_in, n_classes) for multinomial; (d_in, 1) for binary
    b: np.ndarray         # (n_classes,) for multinomial; (1,) for binary
    classes: np.ndarray   # (n_classes,) class labels in column order of W
    C: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for X."""
        if self.W.shape[1] == 1:
            # binary case: sklearn stores a single coefficient row for class 1
            logit = X @ self.W[:, 0] + self.b[0]
            return np.where(logit >= 0, self.classes[1], self.classes[0])
        scores = X @ self.W + self.b  # (n, n_classes)
        return self.classes[np.argmax(scores, axis=1)]


def _make_logreg(C: float) -> LogisticRegression:
    # multi_class='auto' -> 'multinomial' for >2 classes, 'ovr' for binary.
    # solver='lbfgs' supports both. max_iter generous to avoid convergence warnings.
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs")


def fit(X: np.ndarray, y: np.ndarray, C: float) -> LogisticProbe:
    model = _make_logreg(C)
    model.fit(X, y)
    # sklearn: coef_ is (n_classes, d_in) for multiclass; (1, d_in) for binary.
    W = model.coef_.T.astype(np.float32)               # (d_in, n_classes_or_1)
    b = model.intercept_.astype(np.float32)            # (n_classes_or_1,)
    return LogisticProbe(W=W, b=b, classes=model.classes_.copy(), C=float(C))


def sweep_C(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Cs: Sequence[float],
) -> tuple[float, list[dict]]:
    """Fit at each C on train, score macro-F1 on val. Return (best_C, results)."""
    results: list[dict] = []
    for C in Cs:
        probe = fit(X_tr, y_tr, C)
        y_hat = probe.predict(X_val)
        macro_f1 = float(f1_score(y_val, y_hat, average="macro"))
        results.append({"C": float(C), "macro_f1": macro_f1})
    best = max(results, key=lambda r: r["macro_f1"])
    return best["C"], results
```

### Step 3.2: Re-export from `linear_trainer/__init__.py`

- [ ] Modify `src/linear_trainer/__init__.py` — replace existing content with:

```python
from linear_trainer.probe import LinearProbe, fit, sweep_alpha
from linear_trainer.mlp_probe import MLPProbe, fit as fit_mlp, sweep as sweep_mlp
from linear_trainer.logistic_probe import LogisticProbe, fit as fit_logistic, sweep_C

__all__ = [
    "LinearProbe", "fit", "sweep_alpha",
    "MLPProbe", "fit_mlp", "sweep_mlp",
    "LogisticProbe", "fit_logistic", "sweep_C",
]
```

### Step 3.3: Implement the training script

- [ ] Create `scripts/train_logistic_probe.py`:

```python
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
# kmer and length sources synthesise X from cached CDS, but still need a
# parquet to pull meta.ensembl_id and y from. Use DNABERT-2's parquet by
# default — meta columns are identical across the two parquets.
META_PARQUET = DATASET_PATHS["dnabert2"]


# ---------- feature loaders ----------

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
        # family5 uses the standard splits; X comes from one of the parquets,
        # y is the categorical 'family' column.
        if dataset in DATASET_PATHS:
            X, _, meta = load_split(name, dataset_path=DATASET_PATHS[dataset])
        else:
            # kmer / length: still need meta + y from the standard split,
            # but synthesise X from cached CDS.
            _, _, meta = load_split(name, dataset_path=META_PARQUET)
            if dataset == "kmer":
                X = _kmer_features_for_meta(meta)
            elif dataset == "length":
                X = _length_features_for_meta(meta)
            else:
                raise ValueError(f"unknown dataset: {dataset!r}")
        y = meta["family"].to_numpy()
        return X, y, meta

    # binary tasks
    if task not in BINARY_TASKS:
        raise ValueError(f"unknown task: {task!r}")
    if dataset in DATASET_PATHS:
        X, y, meta = load_binary_split(task, name, dataset_path=DATASET_PATHS[dataset])
    else:
        # kmer/length on a binary subset: load via DNABERT-2 parquet for meta+y,
        # then overwrite X.
        _, y, meta = load_binary_split(task, name, dataset_path=META_PARQUET)
        if dataset == "kmer":
            X = _kmer_features_for_meta(meta)
        elif dataset == "length":
            X = _length_features_for_meta(meta)
        else:
            raise ValueError(f"unknown dataset: {dataset!r}")
    return X, y, meta


# ---------- assertions ----------

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


# ---------- metrics ----------

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


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["dnabert2", "nt_v2", "kmer", "length"])
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
        "feature_source": args.dataset,  # which X was actually used
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
```

### Step 3.4: Smoke-test the script on one cell

- [ ] Run a single cell to confirm the script works end-to-end:

```bash
uv run python scripts/train_logistic_probe.py --dataset dnabert2 --task tf_vs_gpcr
```

Expected: prints C sweep, best C, test metrics. `test_macro_f1` should be > 0.5 (above-chance for a balanced binary task). If it errors with import or shape problems, fix before continuing.

- [ ] Verify a metrics entry was appended:

```bash
uv run python -c "
import json
runs = json.load(open('data/metrics.json'))
last = runs[-1]
assert last['model'] == 'logistic_probe'
assert last['task'] == 'tf_vs_gpcr'
assert last['encoder'] == 'dnabert2'
print('last entry:', last['run_id'], 'macro_f1=', last['test_macro_f1'])
"
```

### Step 3.5: Commit

- [ ] Commit:

```bash
git add src/linear_trainer/logistic_probe.py src/linear_trainer/__init__.py scripts/train_logistic_probe.py
git commit -m "logistic_probe: add multi-class + binary classification probe with C-sweep + matrix script"
```

---

## Task 4: Run the full 15-cell matrix

This task runs the script repeatedly to fill the matrix. No code changes — only execution and inspection.

**The matrix:**

| dataset \ task | family5 | tf_vs_gpcr | tf_vs_kinase |
|---|---|---|---|
| dnabert2 | ✓ | ✓ | ✓ |
| nt_v2 | ✓ | ✓ | ✓ |
| kmer | ✓ | ✓ | ✓ |
| length | ✓ | ✓ | ✓ |
| shuffled (uses NT-v2 X) | ✓ | ✓ | ✓ |

15 runs total.

### Step 4.1: Run the four feature sources × three tasks (12 cells)

- [ ] Run all 12 non-shuffled cells:

```bash
for ds in dnabert2 nt_v2 kmer length; do
  for task in family5 tf_vs_gpcr tf_vs_kinase; do
    echo "=== ${ds} × ${task} ==="
    uv run python scripts/train_logistic_probe.py --dataset "$ds" --task "$task"
  done
done
```

Expected: 12 successful runs. The kmer and length runs re-featurize the CDS each time — they will be slower than the embedding-based runs but should still finish in minutes per cell since the CDS files are cached on disk.

### Step 4.2: Run the three shuffled-label cells

- [ ] Run anti-baseline for each task using NT-v2 X:

```bash
for task in family5 tf_vs_gpcr tf_vs_kinase; do
  echo "=== shuffled × ${task} ==="
  uv run python scripts/train_logistic_probe.py --dataset nt_v2 --task "$task" --shuffle-labels
done
```

Expected: macro-F1 lands near chance:
- `family5`: ~0.10–0.20 (5-way imbalanced)
- `tf_vs_gpcr`: ~0.45–0.55 (balanced binary)
- `tf_vs_kinase`: ~0.45–0.55 (balanced binary)

If shuffled macro-F1 sits *more than 0.05* above the analytical chance, **stop and debug** before trusting the real numbers — the pipeline is leaking.

### Step 4.3: Inspect the results table

- [ ] Pretty-print the 15 new entries in a comparison table:

```bash
uv run python -c "
import json, pandas as pd
runs = json.load(open('data/metrics.json'))
runs = [r for r in runs if r.get('model') == 'logistic_probe']
df = pd.DataFrame([
    {
        'task': r['task'],
        'encoder': r['encoder'],
        'C': r['C'],
        'macro_f1': r['test_macro_f1'],
        'bal_acc': r['test_balanced_accuracy'],
        'acc': r['test_accuracy'],
    }
    for r in runs
])
for task in ['family5', 'tf_vs_gpcr', 'tf_vs_kinase']:
    print(f'\n--- {task} ---')
    sub = df[df['task'] == task].sort_values('macro_f1', ascending=False)
    print(sub.to_string(index=False))
"
```

- [ ] Note the **decision-gate outcome** (per spec section "Decision gate after Phase 4a"):

For each task, compute Δ macro-F1 = (encoder) − (kmer):

```bash
uv run python -c "
import json, pandas as pd
runs = [r for r in json.load(open('data/metrics.json'))
        if r.get('model') == 'logistic_probe' and not r.get('shuffled_labels')]
df = pd.DataFrame([{'task': r['task'], 'encoder': r['encoder'], 'macro_f1': r['test_macro_f1']} for r in runs])
for task in ['family5', 'tf_vs_gpcr', 'tf_vs_kinase']:
    sub = df[df['task'] == task]
    kmer = float(sub[sub['encoder']=='kmer']['macro_f1'].iloc[0])
    print(f'\n{task}: kmer={kmer:.4f}')
    for enc in ['dnabert2', 'nt_v2', 'length']:
        v = float(sub[sub['encoder']==enc]['macro_f1'].iloc[0])
        delta = v - kmer
        flag = '  ** BEATS **' if delta >= 0.02 else ('  (loses)' if delta <= -0.02 else '  (ties)')
        print(f'  {enc:<10s} macro_f1={v:.4f}  delta_vs_kmer={delta:+.4f}{flag}')
"
```

This determines which spec branch you're in: write-up (any "BEATS"), Phase 4b pooling (all "ties"), or stop-and-debug (all "loses").

### Step 4.4: Verify confusion matrices were written

- [ ] Check both 5-way confusion matrices exist:

```bash
ls -la data/confusion_5way_*.json
uv run python -c "
import json
for enc in ('dnabert2', 'nt_v2'):
    cm = json.load(open(f'data/confusion_5way_{enc}.json'))
    print(f'{enc}: classes={cm[\"classes\"]}')
    for row, cls in zip(cm['matrix'], cm['classes']):
        print(f'  {cls:8s} {row}')
"
```

### Step 4.5: Commit results

- [ ] Commit the metrics + confusion matrices:

```bash
git add data/metrics.json data/confusion_5way_dnabert2.json data/confusion_5way_nt_v2.json
git commit -m "run: full 15-cell classification matrix on dnabert2 + nt_v2 + baselines"
```

---

## Task 5: Update findings.md and next_steps.md

The exact wording depends on which decision-gate branch the matrix landed in. The structure below works for any branch; fill in the numbers from Task 4.

**Files:**
- Modify: `findings.md`
- Modify: `next_steps.md`

### Step 5.1: Append a Phase 4 section to `findings.md`

- [ ] Read the current `findings.md` to see the structure.
- [ ] Append a new section before the "Caveats" section:

````markdown
## Phase 4 — Classification reframing

Per the 2026-04-29 classification-pivot spec, we re-ran the question as a *classification* task on the same cached embeddings. Three tasks:

- **5-way:** predict family ∈ {tf, gpcr, kinase, ion, immune} on the full 3244-gene corpus, original 70/15/15 split.
- **tf-vs-gpcr (binary):** 591 of each, downsampled tf, frozen split in `data/binary_tf_gpcr.json`.
- **tf-vs-kinase (binary):** 558 of each, frozen split in `data/binary_tf_kinase.json`.

Probe: logistic regression (L2, multinomial for 5-way), C swept on val over `[1e-2 … 1e3]`, refit on train+val, evaluated once on test. Headline metric: macro-F1.

### Results

[Paste the three-task table from Task 4.3 here — for each task, all five rows
(dnabert2, nt_v2, kmer, length, shuffled) with macro_f1 / bal_acc / acc.]

### Read

[Fill in based on the decision gate in Task 4.3:

- If any encoder beats kmer by ≥ 0.02 macro-F1 on at least one task:
  "Pivoting from regression to classification surfaces signal that the
  regression probe couldn't see. The previous informative-negative read
  was at least partly a target-noise artefact (GenePT anisotropy floor +
  noisy summaries). Encoder X beats the 4-mer baseline by Δ on task Y."

- If all encoders tie kmer (within ±0.02) on every task:
  "Classification reframing did not change the read. Both encoders tie the
  4-mer baseline within ±0.02 macro-F1 across all three tasks. The ceiling
  remains the representation, not the probe or the target. Next: pooling
  re-extraction (Phase 4b)."

- If encoders lose on every task:
  "Pipeline issue — anti-baseline check before any further interpretation."]

### Anti-baseline

Shuffled-label runs landed at [actual numbers] for the three tasks
(analytical chance ≈ 0.10–0.20 / 0.50 / 0.50). All within ±0.05 of
chance — pipeline is honest.
````

### Step 5.2: Update `next_steps.md`

- [ ] Mark the original Phase 4 deliverables as superseded and add the new Phase 4a status. Replace the Phase 4 section with:

```markdown
## Phase 4 — Classification reframing   (deck Weeks 4–5)

Original Phase 4 (retrieval@k, IG attribution, family-classification-on-y_hat,
zero-shot demo, viz, write-up) was superseded by the 2026-04-29 classification-
pivot spec — see `docs/superpowers/specs/2026-04-29-classification-pivot-design.md`.

### Phase 4a — Classification probes   ✅ done

15-cell run matrix (3 tasks × 5 feature sources). Results in `findings.md`
§ "Phase 4 — Classification reframing".

Code: `src/binary_tasks/`, `src/length_baseline/`, `src/linear_trainer/logistic_probe.py`,
`scripts/{make_binary_subsets,train_logistic_probe}.py`.

### Phase 4b — Pooling re-extraction   [STATUS]

[STATUS = "⏳ open" if 4a tied 4-mer; "skipped" if 4a beat 4-mer.]

[If open: re-embed both encoders with five pooling variants in parallel —
mean→mean (the Phase 4a baseline, kept for A/B), mean→D
(`concat[first,last,mean]`, 3× dim), mean→G (D + `max_chunks`, 4× dim),
max→mean (max-pool tokens per chunk, then mean across chunks), and CLS→mean
(CLS token per chunk, then mean across chunks). Re-run the same 15-cell
probe matrix. Strategy is locked in the spec — no learned pooling head, no
other variants in 4b.]

### Phase 4c — Write-up   ⏳ open

- [ ] Results table (15 cells from 4a, plus any 4b cells)
- [ ] 5-way confusion matrix per encoder (already saved as JSON)
- [ ] Discussion in the appropriate framing per the decision gate
```

Also remove the old Phase 4 bullet list (retrieval@k, family-classification, zero-shot demo, visualisation, IG, write-up) — it's superseded.

### Step 5.3: Commit the docs

- [ ] Commit:

```bash
git add findings.md next_steps.md
git commit -m "docs: phase 4a results + updated next_steps after classification pivot"
```

---

## Decision gate at the end of Task 4

Per the spec, after Task 4.3 the project is in one of three branches:

| Outcome | Action |
|---|---|
| Any encoder beats 4-mer by Δ macro-F1 ≥ 0.02 on at least one task | **Stop here.** Task 5 wraps up Phase 4a as the headline result. Phase 4b is skipped. The write-up frames the pivot as having surfaced signal that regression hid. |
| Both encoders tie 4-mer (within ±0.02) on every task | **Trigger Phase 4b** (separate spec/plan). Strategy is locked in the spec § "Pooling deferred — strategy locked for Phase 4b": re-embed both encoders with five pooling variants in parallel — `mean→mean` (Phase 4a baseline), `mean→D = concat[first,last,mean]` (3× dim), `mean→G = concat[first,last,mean,max]` (4× dim), `max→mean` (max-pool tokens then mean across chunks), and `CLS→mean` (CLS token per chunk then mean across chunks). Re-run the same probe matrix. No learned pooling head. |
| Encoder loses to 4-mer on every task | **Stop and debug** before continuing. Re-check anti-baseline, then verify the cached embeddings haven't been corrupted (regenerate from `scripts/run_encoder.py` and `scripts/run_nt_v2_encoder.py` if needed). |

Phase 4b is **out of scope for this plan** — if needed, it gets its own spec + plan.

---

## Sanity checklist for the whole sprint

After Task 5, all of the following should be true:

- [ ] `data/binary_tf_gpcr.json` and `data/binary_tf_kinase.json` exist, both with `n_per_class > 500` and three disjoint splits whose union equals `2 * n_per_class`.
- [ ] `data/metrics.json` has 15 new entries with `model == "logistic_probe"`.
- [ ] `data/confusion_5way_dnabert2.json` and `data/confusion_5way_nt_v2.json` exist with 5×5 matrices over `["gpcr","immune","ion","kinase","tf"]` (alphabetical).
- [ ] All three shuffled-label runs land within ±0.05 of analytical chance.
- [ ] `findings.md` has a Phase 4 section with the results table and a clear read.
- [ ] `next_steps.md` reflects the actual decision-gate branch.
- [ ] All five commits are on the `classification-pivot` branch (not `main`).
