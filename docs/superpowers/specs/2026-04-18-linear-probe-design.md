# Linear Probe Training — Design

Design for phase 2 (splits) and phase 3 (linear probe) of the DNA-to-text project. Downstream of `data/dataset.parquet`, which already contains `x (DNABERT-2, 768)` and `y (GenePT, 1536)` for every gene.

Reference documents:
- `framework.md` — experiment structure and hypothesis
- `next_steps.md` — phase-by-phase plan
- `project.md` — project overview

## Scope

This spec covers two new packages:

1. `src/splits/` — frozen stratified 70/15/15 split + loader helper.
2. `src/linear_trainer/` — Ridge probe (`W: R^768 -> R^1536`) with alpha sweep.

Each lands as a separate commit.

**Out of scope (deferred to later specs):**
- Zero-shot carve-out (phase 2 bullet, deferred — needs empirical grounding on summary distribution).
- 4-mer k-mer baseline (phase 3 bullet — separate package later).
- Shuffled-`y` anti-baseline run (uses the splits loader, but is its own script).
- Full metrics module: retrieval@k, family-classification accuracy (later phase).
- Zero-shot demo, visualisation, interpretability, write-up (phases 4–7).

## Package 1 — `src/splits/`

### Purpose
Produce a frozen train/val/test split stratified by `family` and expose a loader so every downstream script reads the same arrays.

### Layout
```
src/splits/
  __init__.py
  make_splits.py     # build logic
  loader.py          # load_split(name)
scripts/
  make_splits.py     # CLI entrypoint
```

### Artefact
`data/splits.json`:
```json
{
  "train": ["ENSG...", ...],
  "val":   ["ENSG...", ...],
  "test":  ["ENSG...", ...],
  "seed":  42,
  "stratify": "family"
}
```
Frozen once written. All downstream scripts depend on this file's determinism.

### API

```python
# src/splits/loader.py
def load_split(name: Literal["train", "val", "test"]
               ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns (X, Y, meta) where:
      X    : (n, 768)  — DNABERT-2 embeddings, row-aligned with meta
      Y    : (n, 1536) — GenePT embeddings, row-aligned with meta
      meta : DataFrame with columns [ensembl_id, symbol, family, summary]
    """

def load_shuffled_y(name: Literal["train", "val", "test"],
                    seed: int = 42
                    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Same as load_split but Y is a seeded permutation within the split.
    Used by the anti-baseline (shuffled-y control run).
    """
```

### Algorithm (`src/splits/make_splits.py`)
1. Load `data/dataset.parquet`.
2. Stratified split via `sklearn.model_selection.train_test_split`:
   - First split 70/30 with `stratify=family, random_state=42`.
   - Then split the 30% 50/50 (stratified again) → 15/15.
3. Assert `train ∪ val ∪ test == full_corpus` and all three are pairwise disjoint. Fail loud if not.
4. Write `data/splits.json`.

### CLI (`scripts/make_splits.py`)
- No arguments. Reads `data/dataset.parquet`, writes `data/splits.json`. Idempotent: re-running with the same seed produces the same file.

## Package 2 — `src/linear_trainer/`

### Purpose
Fit the primary probe from `framework.md`: one-layer regularised linear map `W: R^768 -> R^1536` via Ridge regression.

### Layout
```
src/linear_trainer/
  __init__.py
  probe.py           # LinearProbe class + fit + sweep_alpha
scripts/
  train_probe.py     # CLI: sweep alpha, refit, save artefact + metrics
```

### API

Class for the trained artefact; free functions for training (decision C from brainstorming).

```python
# src/linear_trainer/probe.py

@dataclass
class LinearProbe:
    W: np.ndarray        # (768, 1536)
    b: np.ndarray        # (1536,)
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X @ W + b"""

    def save(self, path: str | Path) -> None:
        """np.savez(path, W=W, b=b, alpha=alpha)"""

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe": ...

def fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> LinearProbe:
    """Fit sklearn.linear_model.Ridge(alpha).fit(X, Y); return LinearProbe."""

def sweep_alpha(X_tr: np.ndarray, Y_tr: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                alphas: Sequence[float]
                ) -> tuple[float, list[dict]]:
    """
    For each alpha: fit on train, score mean cosine on val.
    Return (best_alpha, [{"alpha": a, "mean_cosine": c} for a in alphas]).
    """
```

### Training procedure (`scripts/train_probe.py`)

1. `X_tr, Y_tr, _ = load_split("train")`
   `X_val, Y_val, _ = load_split("val")`
   `X_te,  Y_te,  _ = load_split("test")`
2. `best_alpha, sweep = sweep_alpha(X_tr, Y_tr, X_val, Y_val, alphas=[1e-2, 1e-1, 1, 10, 100, 1000])`
3. Refit at `best_alpha` on `np.vstack([X_tr, X_val])`, `np.vstack([Y_tr, Y_val])`.
4. Evaluate on test: compute mean cosine, median cosine, macro R².
5. Write `data/probe.npz` via `LinearProbe.save`.
6. Append a run entry to `data/metrics.json`.

### Inline cosine for alpha selection
Mean cosine is computed inline in `sweep_alpha` and in the test-evaluation step:
```python
num = (y_hat * y).sum(-1)
den = np.linalg.norm(y_hat, axis=-1) * np.linalg.norm(y, axis=-1)
cos = num / den          # shape (n,)
```
No separate `metrics.py` module this round. A proper metrics package lands when retrieval@k and family-classification arrive.

### Metrics artefact (`data/metrics.json`)

File is a JSON array, appended to per run. Each entry:
```json
{
  "run_id": "probe_<timestamp>",
  "timestamp": "2026-04-18T...",
  "model": "linear_probe",
  "alpha": 10.0,
  "alpha_sweep": [{"alpha": 0.01, "mean_cosine": ...}, ...],
  "test_mean_cosine": ...,
  "test_median_cosine": ...,
  "test_r2_macro": ...
}
```
Future runs (baseline, shuffled-y, etc.) share the same schema so all runs live in one file.

## Dependencies

Add to `pyproject.toml`:
- `scikit-learn>=1.4`

## Sanity assertions

Not a test suite — assertions inside the scripts that fail loud on pipeline breakage.

- `make_splits.py`: splits are disjoint and their union equals the corpus.
- `train_probe.py`:
  - `W.shape == (768, 1536)`, `b.shape == (1536,)`
  - `test_mean_cosine > 0` (catches catastrophic breakage; real correctness gate is the later shuffled-`y` anti-baseline scoring near zero).

## Commit plan

**Commit 1 — splits package**
- Adds `src/splits/{__init__.py, make_splits.py, loader.py}`
- Adds `scripts/make_splits.py`
- Adds `scikit-learn>=1.4` to `pyproject.toml`
- Produces `data/splits.json`
- Message: `splits: add stratified 70/15/15 splitter and load_split helper`

**Commit 2 — linear_trainer package**
- Adds `src/linear_trainer/{__init__.py, probe.py}`
- Adds `scripts/train_probe.py`
- Produces `data/probe.npz` and initial entry in `data/metrics.json`
- Message: `linear_trainer: add Ridge probe + alpha sweep training`

## What lands later (not this spec)

- Zero-shot carve-out (decide criterion after looking at summary-length distribution).
- `src/kmer_baseline/` — 4-mer frequency featuriser + baseline train script (same Ridge, same metrics schema).
- Shuffled-`y` anti-baseline run (tiny script, consumes `load_shuffled_y`).
- `src/metrics/` — retrieval@k, family-classification accuracy, R² helpers.
- Phases 4–7 (zero-shot demo, viz, interp, write-up).
