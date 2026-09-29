# Stage 5: Bootstrap Test-Set Confidence Intervals

`scripts/bootstrap_test_uncertainty.py` produces 95% percentile confidence
intervals for every headline metric reported in the paper, plus per-class F1
for the family-classification cells. Output is cached at
`data/bootstrap_metrics.json` and rendered into the abstract, Methods §Probes,
and Results §3.1–§3.2.

## Pipeline placement

This is Stage 5 of the repository pipeline. Run it after the Stage 3 probe and
baseline cells and the Stage 4 TSS cells have been run in this checkout. It
fits nothing: it reads each headline cell's record from
`data/metrics_homology.json` (chosen on validation by `scripts/headline_cells.py`),
loads the test predictions that run stored (`pred_file` under
`outputs/predictions/`, not tracked), and writes `data/bootstrap_metrics.json`.
Records from before stored predictions existed have no `pred_file`; the script
stops with "no stored predictions; rerun the cell".

Sample input and output shapes are tracked in
`samples/stage5_bootstrap_input.json` and
`samples/stage5_bootstrap_output.json`.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/bootstrap_test_uncertainty.py` | Rescore each headline record's stored, hash-verified test predictions, bootstrap the test metrics, and write confidence intervals. |
| `scripts/headline_cells.py` | Validation-selected headline cells (explicit pool lists), read from `data/metrics_homology.json`. |
| `data/bootstrap_metrics.json` | Cached 1,000-run bootstrap output used by the report. |
| `outputs/predictions/<metrics stem>/` | Stored test predictions, one content-addressed `.npz` per record (not tracked). |
| `samples/stage5_bootstrap_input.json` | Tiny summary of bootstrap inputs and settings. |
| `samples/stage5_bootstrap_output.json` | Tiny excerpt of bootstrap confidence interval output. |

## What it does

For each headline cell:

1. **Load** the record's stored test predictions (`pred_file`) and verify them
   against `pred_sha256`. Nothing is refitted: a refit can land on a different
   model (another machine, thread count or `max_iter`), and the interval would
   then no longer bracket the reported value.
2. **Check** that the predictions reproduce the record's test value exactly;
   the script stops otherwise.
3. **Resample** the test split 1,000 times with replacement, recompute the
   metric on each resample, and take the 2.5th / 97.5th percentiles as the
   95% CI.

Resampling is **stratified by family** for classification (so each bootstrap
draw has the same per-class size as the real test set) and **i.i.d. by gene**
for regression.

Per-class F1 is computed once on the full test predictions — no bootstrap.

## What it is not

The bootstrap captures **test-composition sampling uncertainty only**. It
does not reflect:

- Hyperparameter sensitivity (no re-sweep of `C` or `α`).
- Train/val split-seed variability (the split is fixed).
- Encoder-side variability (embeddings are taken as given).

The report Methods section states this scope explicitly.

## Cells covered

The cells come from `scripts/headline_cells.py`, so they follow the
validation-selected picks rather than a fixed list:

- **Classification (`HEADLINE_CLS`, `HEADLINE_CLS_TSS`):** the composition
  comparators (4-mer, codon, amino-acid 1/2/3-mer), each DNA encoder's
  validation-selected pool, ESM-2 650M, the shuffled-label anti-baseline, and
  on the TSS window the TSS 4-mer plus each encoder's selected pool.
  Reported: macro-F1 and Cohen's kappa (point and CI), plus per-class F1.
- **Regression (`HEADLINE_REG`, `HEADLINE_REG_TSS`):** the same comparators,
  each encoder's selected pool and ESM-2 650M, on CDS and TSS. Reported:
  macro-R^2 across the 1,536 GenePT dimensions (point and CI).
- **Paired differences (`--paired`):** `PAIRED_CLS` / `PAIRED_REG` resample the
  same test genes for both cells of each comparison; CDS vs TSS pairs use the
  genes common to both test sets.

## Running it

```bash
uv run python scripts/bootstrap_test_uncertainty.py
```

Optional flags:

- `--n-iters N`: number of bootstrap iterations (default `1000`).
- `--seed S`: RNG seed for reproducibility (default `42`).
- `--out PATH`: override output path (default `data/bootstrap_metrics.json`).
- `--paired`: also compute paired-difference CIs.

## Output schema

```json
{
  "n_iters": 1000,
  "seed": 42,
  "classification": {
    "<cell_name>": {
      "n_test": 487,
      "macro_f1_point": 0.82,
      "macro_f1_ci95": [0.77, 0.87],
      "kappa_point": 0.79,
      "kappa_ci95": [0.74, 0.83],
      "per_class_f1": {"tf": 0.93, "gpcr": 0.94, ...},
      "n_iters": 1000
    },
    ...
  },
  "regression": {
    "<cell_name>": {
      "n_test": 487,
      "r2_macro_point": 0.21,
      "r2_macro_ci95": [0.18, 0.24],
      "n_iters": 1000
    },
    ...
  }
}
```

## What the results show

These readings are from the May 2026 protocol and will be re-checked after the
camera-ready recompute.

- **NT-v2 vs 4-mer (classification):** CIs are non-overlapping on both
  macro-F1 and κ. The encoder gain is not a test-split artifact.
- **DNABERT-2 vs 4-mer (regression):** CIs are non-overlapping on R².
- **Per-class F1:** the encoder advantage is concentrated in the minority
  classes — immune +0.33, ion +0.29 over the 4-mer baseline — rather than
  spread evenly across families.

The `shuffled` and `shuffled_y` rows act as anti-baselines: their CIs
straddle chance for classification (~0.20 macro-F1) and zero for
regression, confirming the protocol is working.
