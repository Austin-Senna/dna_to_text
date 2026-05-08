# Bootstrap test-set confidence intervals

`scripts/bootstrap_test_uncertainty.py` produces 95% percentile confidence
intervals for every headline metric reported in the paper, plus per-class F1
for the family-classification cells. Output is cached at
`data/bootstrap_metrics.json` and rendered into the abstract, Methods §Probes,
and Results §3.1–§3.2.

## What it does

For each headline cell (12 total: 6 classification + 6 regression):

1. **Refit** the probe at the recorded best hyperparameter on `train + val`,
   matching the original protocol exactly. Hyperparameters come from the
   latest matching entry in `data/metrics.json`.
2. **Predict** once on the held-out test split.
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

The Methods section states this scope explicitly.

## Cells covered

### Classification (`HEADLINE_CLS`)

| Cell name | Dataset | C | Notes |
|---|---|---|---|
| `kmer` | 4-mer baseline | 1000.0 | k-mers re-featurised from CDS at runtime |
| `dnabert2_meanD` | DNABERT-2, meanD pooling | 10.0 | |
| `nt_v2_meanD` | NT-v2, meanD pooling | 1.0 | headline encoder |
| `gena_lm_clsmean` | GENA-LM, clsmean pooling | 1.0 | |
| `hyena_dna_meanG` | HyenaDNA, meanG pooling | 10.0 | |
| `shuffled` | NT-v2 with shuffled labels | 100.0 | sanity-check anti-baseline |

Reported metrics: macro-F1 + Cohen's κ (both point estimate and CI), plus
per-class F1 across {tf, gpcr, kinase, ion, immune}.

### Regression (`HEADLINE_REG`)

| Cell name | Dataset | α | Notes |
|---|---|---|---|
| `kmer` | 4-mer baseline | 0.01 | |
| `dnabert2_meanG` | DNABERT-2, meanG pooling | 10.0 | headline encoder for regression |
| `nt_v2_meanmean` | NT-v2, meanmean pooling | 10.0 | |
| `gena_lm_meanmean` | GENA-LM, meanmean pooling | 100.0 | |
| `hyena_dna_specialmean` | HyenaDNA, specialmean pooling | 1.0 | |
| `shuffled_y` | NT-v2 with shuffled GenePT vectors | 1000.0 | anti-baseline |

Reported metric: macro-R² across the 1,536 GenePT dimensions (point estimate
and CI).

## Running it

```bash
uv run python scripts/bootstrap_test_uncertainty.py
```

Optional flags:

- `--n-iters N` — number of bootstrap iterations (default `1000`).
- `--seed S` — RNG seed for reproducibility (default `42`).
- `--out PATH` — override output path (default `data/bootstrap_metrics.json`).

The 4-mer cells re-featurise CDS sequences from `data/sequences/` at runtime;
all encoder cells load embeddings from `data/dataset_<encoder>_<pooling>.parquet`.

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

- **NT-v2 vs 4-mer (classification):** CIs are non-overlapping on both
  macro-F1 and κ. The encoder gain is not a test-split artifact.
- **DNABERT-2 vs 4-mer (regression):** CIs are non-overlapping on R².
- **Per-class F1:** the encoder advantage is concentrated in the minority
  classes — immune +0.33, ion +0.29 over the 4-mer baseline — rather than
  spread evenly across families.

The `shuffled` and `shuffled_y` rows act as anti-baselines: their CIs
straddle chance for classification (~0.20 macro-F1) and zero for
regression, confirming the protocol is working.
