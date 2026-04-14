# Framework

How the experiment is structured after the data pipeline. The input is `data/dataset.parquet` with `symbol, ensembl_id, family, summary, x (DNABERT-2 768d), y (GenePT 1536d)`.

## Hypothesis

A linear map `W: R^768 -> R^1536` can transport DNABERT-2 CDS embeddings into GenePT text-embedding space well enough to recover gene function. If so, DNABERT-2 has implicitly encoded functional semantics from raw nucleotide syntax.

## Data split

- Stratified by `family` so every family appears in train/val/test.
- 70 / 15 / 15 split, seed-fixed.
- Zero-shot set: a small hold-out of poorly/uncharacterised genes (summary absent or generic) pulled from outside the family-balanced corpus. Used only for the qualitative demo, not for tuning.

## Models

### Probe (primary)

Ridge Regression, multi-output: `y_hat = X W + b`, one regularised linear layer.

- Implementation: `sklearn.linear_model.Ridge` (multi-output by default).
- Hyperparameter: `alpha` chosen on val split via grid over `[1e-2, 1e-1, 1, 10, 100, 1000]`.
- No nonlinearity on purpose — the whole point is to probe *linear* alignment.

### Baseline (control)

Same Ridge Regression on 4-mer frequency vectors of the CDS (256-d for DNA). Tells us how much of the signal comes from DNABERT-2's learned representation vs raw sequence composition.

### Anti-baseline (sanity)

Ridge on shuffled `y` — should collapse to near-zero R². Catches pipeline leakage.

## Metrics

Computed on the held-out test split.

| Metric | Definition | Why |
|---|---|---|
| Cosine similarity (mean, median) | `cos(y_hat_i, y_i)` per gene | Direction in embedding space is what downstream tasks care about. |
| R² (macro across 1536 dims) | `sklearn.metrics.r2_score(..., multioutput="uniform_average")` | Standard regression score. |
| Retrieval@k | For each test `y_hat_i`, rank all real summaries; is `summary_i` in top-k? | Task-level: does the projection land near the right gene? |
| Family classification accuracy | Train logistic regression on `y_hat` to predict `family`; compare to same classifier on real `y` | How much class-level structure survives the projection. |

All metrics reported for: probe, k-mer baseline, shuffled anti-baseline.

## Zero-shot demo

For each gene in the zero-shot set:

1. Fetch CDS, embed with DNABERT-2 → `x`.
2. Project: `y_hat = x W + b`.
3. Nearest neighbours in real GenePT space → predicted functional family (majority vote of top-k).
4. Report predicted family, top-5 neighbour genes, cosine to centroid of each family.

Success criterion is qualitative: the predicted family should be biologically plausible given any available annotation.

## Visualisation

Three 2-D plots, same points, coloured by `family`:

1. DNA space (`x`) — PCA and UMAP.
2. Text space (`y`) — PCA and UMAP.
3. Projected space (`y_hat`) — PCA and UMAP on test split.

Useful if (2) and (3) show the same cluster topology; damning if (1) and (3) do.

## Interpretability

Captum Integrated Gradients against DNABERT-2, with the probe stacked on top so the target is a scalar projection into GenePT space (e.g. cosine to the family centroid).

- Attribute token-level importance over the CDS.
- Aggregate back to nucleotide windows; look for motif enrichment inside family-specific attributed regions (kinase domains, zinc-finger motifs, TM helices for GPCRs, etc.).
- This is the "why does it work" half of the story; it is a bonus, not a gate on success.

## Success / failure criteria

- **Positive result:** probe beats 4-mer baseline on cosine and retrieval@k, and family clusters are visibly preserved in projected space.
- **Informative negative:** probe matches or barely beats 4-mer baseline → DNABERT-2's extra capacity is not functional, it is compositional. Still publishable as a limits-of-genomic-LLMs finding.
- **Pipeline bug:** anti-baseline (shuffled `y`) scores non-trivially. Stop and debug before drawing any conclusion.

## Repo layout for this phase

```
src/dna_to_text/
  probe.py          # Ridge fit + predict + save W,b
  baseline.py       # 4-mer featuriser + Ridge
  metrics.py        # cosine, R², retrieval@k, family acc
  splits.py         # stratified train/val/test; shuffled control
  interpret.py      # Captum IG over DNABERT-2 + stacked probe
  viz.py            # PCA / UMAP plots
scripts/
  train_probe.py    # reads dataset.parquet, writes probe.npz + metrics.json
  train_baseline.py # same shape, k-mer features
  zero_shot.py      # demo on uncharacterised genes
  make_plots.py     # figures for the write-up
data/
  splits.json       # frozen train/val/test ensembl_ids
  probe.npz         # W, b, alpha
  metrics.json      # all runs, all metrics
  figures/          # PNG / PDF outputs
```
