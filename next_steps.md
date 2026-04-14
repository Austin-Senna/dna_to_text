# Next Steps

Phase 1 (data + embeddings) is done: `data/dataset.parquet` exists with `x (768)` and `y (1536)` for every gene across the 5 families. Everything below is downstream of that file.

## Phase 2 — Splits and controls

Blocks everything else. Must land before modelling starts.

- [ ] `src/dna_to_text/splits.py`: stratified 70/15/15 on `family`, seeded. Dump `data/splits.json` as `{train: [ensembl_id...], val: [...], test: [...]}`.
- [ ] Carve out zero-shot set (~20–30 genes). Candidates: genes with empty / very short summaries, or outside the 5 families. Save to `data/zero_shot.parquet`.
- [ ] Helper: `load_split(name) -> (X, Y, meta)` so every downstream script reads the same arrays.
- [ ] Shuffled-`y` control set ready to use as a sanity run in phase 3.

## Phase 3 — Probe + baseline + sanity

Primary modelling work. Probe, baseline, and shuffled control can run in parallel once phase 2 lands.

- [ ] `src/dna_to_text/probe.py`: thin wrapper around `sklearn.linear_model.Ridge` with `.fit(X, Y)`, `.predict(X)`, `.save(path)`, `.load(path)`.
- [ ] `scripts/train_probe.py`:
  - Load splits.
  - Sweep `alpha in [1e-2, 1e-1, 1, 10, 100, 1000]` on val, pick best by mean cosine.
  - Refit on train+val at chosen alpha.
  - Write `data/probe.npz` (W, b, alpha) and append to `data/metrics.json`.
- [ ] `src/dna_to_text/baseline.py`: 4-mer frequency featuriser over CDS (256-d, L1-normalised).
- [ ] `scripts/train_baseline.py`: same shape as `train_probe.py`, same metrics schema.
- [ ] `src/dna_to_text/metrics.py`: cosine (mean, median), macro R², retrieval@1/5/10, family-classification accuracy.
- [ ] Shuffled-`y` control run — should score near zero. If not, pipeline is leaking; stop and debug.
- [ ] Gate: probe beats 4-mer baseline on cosine and retrieval@k?

## Phase 4 — Zero-shot demo

Depends on phase 3 (needs trained probe).

- [ ] `scripts/zero_shot.py`: load `zero_shot.parquet` + `probe.npz`, project `x -> y_hat`, k-NN against the training `y` matrix, majority-vote family among top-5, report neighbour symbols + cosines.
- [ ] Pick 3–5 illustrative genes for the write-up.
- [ ] Emit a markdown table of predictions → neighbours for the report.

## Phase 5 — Visualisation

Can start as soon as phase 3 has a trained probe; independent of phase 4.

- [ ] `src/dna_to_text/viz.py`: PCA and UMAP helpers that take an `(N, D)` matrix + labels and return a figure.
- [ ] `scripts/make_plots.py`: DNA space (`x`), text space (`y`), projected space (`y_hat` on test). Two projections (PCA, UMAP) × three spaces = 6 panels, consistent colour map across panels. Save to `data/figures/`.

## Phase 6 — Interpretability

Depends on phase 3 probe + phase 5 plots for context.

- [ ] `src/dna_to_text/interpret.py`: stack DNABERT-2 + fixed probe so the output is a scalar (cosine to a chosen family centroid in GenePT space). Wrap with Captum IG.
- [ ] Pick 1–2 genes per family; attribute at the token level; map tokens back to nucleotide spans.
- [ ] Look for motif enrichment in high-attribution spans (manual inspection first; proper enrichment test is bonus).
- [ ] One figure per family showing attribution over the CDS.

## Phase 7 — Write-up

Final phase. Consumes outputs of 3–6.

- [ ] Intro (condense from `project.md`).
- [ ] Methods (point at `framework.md` and `src/data_loader/pipeline.md`).
- [ ] Results table: probe vs baseline vs shuffled control.
- [ ] Figures: UMAP panel, zero-shot table, one IG figure.
- [ ] Discussion: positive / informative-negative framing depending on the numbers.

## Dependency graph

```
[2 splits] ──► [3 probe + baseline + sanity] ──┬─► [4 zero-shot] ─┐
                                                │                  │
                                                └─► [5 viz] ──► [6 interp] ──► [7 write-up]
```

## Open questions

- Mean-of-chunks pooling (current) vs CLS / max-pool variants? Decide after first probe run; only revisit if cosine plateaus well below baseline.
- 5 families enough resolution, or add sub-family labels for a harder classification test? Defer until cluster separation is visible.
- Version `dataset.parquet` if pooling changes? For now, no — single frozen version in `data/`.
