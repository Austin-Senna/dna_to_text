# Classification Pivot — Design

Sprint design for Phase 4 of the DNA-to-text project, replacing the original Phase 4 deliverables (retrieval@k, IG attribution, etc.) in `next_steps.md`. Driven by Hayden's 2026-04-20 feedback after the Phase 3 informative negative:

1. Make this an easier problem — pick a specific subset of genes with different functions, predict something simpler.
2. Don't run Integrated Gradients on already-bad results.
3. Chunks don't have meaning — try something other than averaging the embeddings.

Reference documents:
- `findings.md` — Phase 3 results (linear + MLP probes, both encoders tie 4-mer baseline within 0.002 cosine).
- `framework.md` — experiment structure.
- `next_steps.md` — phase log; this spec supersedes its Phase 4 bullets.

## Hypothesis pivot

Replace the regression hypothesis ("a linear map can transport DNA embeddings into GenePT space") with a classification hypothesis:

> A pretrained DNA encoder's CDS embedding carries enough functional signal to classify a gene's family above a 4-mer composition baseline.

Classification sidesteps two confounds that hurt regression:
- GenePT's anisotropy floor (mean-Y has cosine ≈ 0.91 with any real GenePT vector — cosine was a misleading headline).
- Per-gene summary noise — the regression target was the text embedding of an NCBI summary, and many summaries are short or boilerplate. Classification reduces the target to a discrete label that doesn't depend on summary quality.

## Tasks

Three classification tasks, all on the existing 3244-gene corpus, all consuming the cached embeddings in `data/dataset.parquet` (DNABERT-2, 768-d) and `data/dataset_nt_v2.parquet` (NT-v2, 512-d).

| Task | Classes | n per class | Total | Train / val / test |
|---|---|---:|---:|---|
| **5-way** | tf, gpcr, kinase, ion, immune | 1743 / 591 / 558 / 198 / 154 | 3244 | 2270 / 487 / 487 (existing split) |
| **binary tf vs gpcr** | tf (downsampled), gpcr | 591 / 591 | 1182 | 826 / 178 / 178 |
| **binary tf vs kinase** | tf (downsampled), kinase | 558 / 558 | 1116 | 781 / 167 / 168 |

The two binary tasks test the same hypothesis under cleaner balance: one on biologically *very distinct* families (nuclear DNA-binder vs 7TM membrane receptor — a strong test where the 4-mer baseline is also expected to be strong), one on intermediate distinctness (nuclear vs cytoplasmic kinase). The 5-way task gives the headline number for the write-up.

### Why two binary tasks instead of one

A single binary task is fragile to family choice: tf-vs-gpcr is *biologically* easy (codon composition diverges sharply between membrane and nuclear proteins) which makes it *too easy* for the 4-mer baseline. tf-vs-kinase is harder for k-mer (both intracellular soluble) but biologically less obvious. Running both gives a contrast — if the encoder beats k-mer on one but not the other, the difference itself is informative.

### Pooling deferred

Hayden's "chunks don't have meaning" critique is parked. Run all three tasks on the *existing* mean-pooled embeddings first. If classification ties 4-mer (same outcome as regression), invest in alternative pooling. If classification beats 4-mer with mean-pool, pooling was not the bottleneck and the negative-result framing already had escape valves we can now close.

This is a sequencing decision: pooling re-extraction is a 1–2 day commitment per encoder, and it's only worth it if the simpler intervention fails.

## Probes and baselines

For each (encoder, task) cell, train logistic regression with L2; sweep `C ∈ [1e-2, 1e-1, 1, 10, 100, 1000]` on val, refit on train+val, evaluate once on test.

For each task, also run three baselines (encoder-independent or below-encoder):

1. **4-mer + logistic.** Same 256-d L1-normalised 4-mer features from `src/kmer_baseline/featurizer.py`, swap Ridge → LogisticRegression. The "did the encoder beat raw composition" gate.
2. **Shuffled-label anti-baseline.** Permute y in train+val, evaluate on real test labels. Should land near chance: ~50% accuracy / ~0.50 macro-F1 for balanced binary, ~20% accuracy / ~0.10–0.20 macro-F1 for the imbalanced 5-way. Run once per task (pick one X — NT-v2, since the result is X-agnostic). Pipeline sanity gate.
3. **CDS length only.** Single feature: `len(cds)` → logistic. Catches the embarrassing "encoder is just a length proxy" failure. Length distributions differ across families (kinase domains short, ion channels long) so this baseline may already be informative.

No MLP probe at all. Phase 3 already showed that MLP depth/width doesn't move the number on these encoders (1–3 hidden layers all tied the linear probe within ±0.002 cosine). Re-running an MLP diagnostic on the classification reframing would not surprise. If logistic ties 4-mer, the answer is pooling re-extraction (Phase 4b), not more probe capacity.

## Run matrix

```
                       5-way   tf-vs-gpcr   tf-vs-kinase
DNABERT-2 logistic       ✓         ✓             ✓
NT-v2 logistic           ✓         ✓             ✓
4-mer logistic           ✓         ✓             ✓
shuffled-label           ✓         ✓             ✓
length-only              ✓         ✓             ✓
                                                = 15 runs
```

All 15 results land in `data/metrics.json` under a new `model` schema (see Metrics artefact below).

## Frozen subsets

The two binary tasks downsample tf to match the smaller class. The downsampled subset must be **deterministic** so every probe + baseline + anti-baseline trains on the same genes. Fix once, store as JSON, never re-roll.

Files:

```json
// data/binary_tf_gpcr.json
{
  "task": "tf_vs_gpcr",
  "seed": 42,
  "tf_ensembl_ids":   ["ENSG...", ...],   // 591 entries, deterministic sample of full tf set
  "gpcr_ensembl_ids": ["ENSG...", ...],   // 591 entries, all gpcr
  "split": {
    "train": ["ENSG...", ...],
    "val":   ["ENSG...", ...],
    "test":  ["ENSG...", ...]
  }
}
```

`data/binary_tf_kinase.json` follows the same schema with 558 of each.

The split is computed once at subset-creation time, stratified 70/15/15 on the binary label, and frozen alongside the subset selection. This avoids the bug of "subset deterministic, split non-deterministic across runs."

## Metrics

Computed on the held-out test split per task.

| Metric | Why |
|---|---|
| **macro-F1** (headline) | Robust to class imbalance (5-way is 1743 / 154 at the extremes). |
| **balanced accuracy** | Sanity companion — average per-class recall. |
| **per-class accuracy** | Goes in the discussion; tells us *which* families confuse. |
| **plain accuracy** | Also recorded — useful for binary tasks where it equals balanced accuracy by construction. |

Confusion matrix (5-way only) saved as a separate artefact for the write-up.

## Scope

### Packages added or extended

1. **`src/binary_tasks/`** — new package. Builds and loads the two binary subsets.
   - `make_subsets.py` — deterministic downsample + stratified split, writes the two JSON files.
   - `loader.py` — `load_binary_split(task, name, dataset_path)` returning `(X, y_binary, meta)`.
2. **`src/linear_trainer/logistic_probe.py`** — new module alongside `probe.py`. Logistic regression sweep with the same fit/refit/eval discipline as the Ridge probe. Multi-class via `multi_class="multinomial"`, binary via plain logistic.
3. **`src/kmer_baseline/featurizer.py`** — already exists; reused unchanged. Just feed its output into logistic instead of Ridge.
4. **`src/length_baseline/`** — new tiny package. Single function `cds_length_features(meta) -> np.ndarray`. Reads cached FASTAs in `data/sequences/`.

### Scripts added

- `scripts/make_binary_subsets.py` — CLI for `src/binary_tasks/make_subsets.py`. Idempotent.
- `scripts/train_logistic_probe.py` — CLI flags: `--dataset {dnabert2,nt_v2,kmer,length}`, `--task {family5,tf_vs_gpcr,tf_vs_kinase}`. One script handles every cell of the run matrix.
- `scripts/train_anti_baseline_classification.py` — shuffled-label variant of the above. Could be a flag on the main script (`--shuffle-labels`) instead — see Decision below.

### Decision: `--shuffle-labels` flag vs separate script

Make it a flag on `train_logistic_probe.py`. The Ridge anti-baseline got its own script in Phase 3, but for classification the only difference is one line (label permutation), and a flag makes it harder to forget the anti-baseline for a new task. Trade-off: slightly more conditional code in one script vs five new scripts.

### Out of scope (deliberately)

- **Pooling re-extraction.** Conditional on Phase 4a results. If needed, becomes its own spec.
- **Captum Integrated Gradients.** Hayden's call: don't gradient on bad results. Revisit only if a probe meaningfully clears 4-mer.
- **Retrieval@k, family-classification-on-y_hat.** Superseded by direct classification on x. The "predict GenePT then classify" detour was a regression-era idea.
- **Visualisation (PCA / UMAP).** Optional Phase 4c deliverable; only if the write-up needs a figure. Not blocking.
- **Third encoder (HyenaDNA / Caduceus / GENA-LM).** Two encoders already give converging evidence.
- **Re-running regression probes.** The Phase 3 numbers stand; we are pivoting, not re-running.

## Metrics artefact (`data/metrics.json`)

Existing file is a JSON array, append-only. New entries follow this schema (additive — old entries stay valid):

```json
{
  "run_id": "logistic_<encoder>_<task>_<timestamp>",
  "timestamp": "2026-04-29T...",
  "model": "logistic_probe",
  "encoder": "dnabert2 | nt_v2 | kmer | length | shuffled",
  "task": "family5 | tf_vs_gpcr | tf_vs_kinase",
  "C": 10.0,
  "C_sweep": [{"C": 0.01, "macro_f1": ...}, ...],
  "test_macro_f1": ...,
  "test_balanced_accuracy": ...,
  "test_accuracy": ...,
  "test_per_class_accuracy": {"tf": ..., "gpcr": ..., ...},
  "confusion_matrix": [[...], ...]   // 5-way only
}
```

## Sanity assertions

Inside `train_logistic_probe.py`:
- All three splits (train/val/test) are non-empty and pairwise disjoint by `ensembl_id`.
- For binary tasks: per-split class balance within ±5 percentage points of 50/50.
- For 5-way: every class present in train, val, test.
- `test_macro_f1` of the shuffled-label run sits within ±0.05 of the analytical chance value for the task (≈0.50 for binary, ≈0.10–0.20 for 5-way). If it doesn't, pipeline is leaking.

## Decision gate after Phase 4a

Three branches, decided after the run matrix is filled:

"Beats 4-mer" below means a macro-F1 delta ≥ 0.02 (two percentage points absolute). "Ties 4-mer" means within ±0.02 macro-F1.

| Outcome | Branch |
|---|---|
| Either encoder beats 4-mer (Δ macro-F1 ≥ 0.02) on at least one task | **Write up.** Positive result on classification reframing. No pooling work needed. Note in the discussion that the previous regression negative was likely target-noise driven. |
| Both encoders tie 4-mer (within ±0.02 macro-F1) on every task | **Phase 4b: pooling re-extraction.** Pick from the brainstormed menu (max-pool, first-chunk, concat[first,last,mean], or per-token attention head). Re-embed once, re-run the same probe matrix. |
| Encoder loses to 4-mer (Δ ≤ −0.02) on every task | **Stop and debug.** Pipeline issue or feature collapse. Check anti-baseline first, then re-verify the cached embeddings haven't been corrupted. |

The decision gate is *part* of the spec, not deferred. The write-up framing depends on which branch we land in.

## Sprint shape

Roughly one week.

**Phase 4a — Classification on existing embeddings (Days 1–3)**
- Day 1: `make_binary_subsets.py` + frozen JSON artefacts; `length_baseline/`; logistic probe module.
- Day 2: `train_logistic_probe.py` with all four `--dataset` modes and three `--task` modes; smoke test on one cell.
- Day 3: Run the full 15-cell matrix; collect into `metrics.json`; build the headline table.

**Phase 4b — Pooling (conditional, Days 4–6)**
- Triggered only by the "both tie 4-mer" branch above.
- Pick one pooling alternative (cheapest first), re-embed, re-run the matrix.
- Each additional pooling alternative is a half-day cost — budget for at most two.

**Phase 4c — Write-up (Days 6–7)**
- Results table (15 cells, plus any 4b cells).
- 5-way confusion matrix per encoder.
- Discussion: which framing the result lands in (positive / informative-negative / pooling-fixable).
- Update `findings.md` and `next_steps.md`.

## Commit plan

1. **`binary_tasks: add subset builder and loader`** — `src/binary_tasks/`, `scripts/make_binary_subsets.py`, the two JSON artefacts.
2. **`length_baseline: add CDS-length-only feature`** — `src/length_baseline/`.
3. **`logistic_probe: add multi-class + binary classification probe`** — `src/linear_trainer/logistic_probe.py`, `scripts/train_logistic_probe.py` with all flags.
4. **`run: full classification matrix on dnabert2 + nt_v2 + baselines`** — appends 15 entries to `data/metrics.json`, plus any confusion-matrix artefacts.
5. **`docs: phase 4a results in findings.md and next_steps.md`** — write-up of whichever branch the gate lands in.

(4b and its commits only exist if the gate fires.)

## Dependencies

No new pip dependencies. `sklearn.linear_model.LogisticRegression`, `sklearn.metrics.{f1_score, balanced_accuracy_score, confusion_matrix}` are already in scope from Phase 3.

## What lands later (not this spec)

- Pooling re-extraction (conditional Phase 4b — its own spec if triggered).
- Captum IG (only if a probe meaningfully clears 4-mer).
- Visualisation panels (only if the write-up benefits).
- Third encoder.
