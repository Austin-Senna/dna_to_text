# Next Steps

Living project log. Phases 1–3 are done; phase 4 carries the deck's Week 4–5 deliverables; phase 5 collects the open scientific questions surfaced by `findings.md`.

## Phase 1 — Data pipeline   ✅ done   (deck Weeks 1–2)

- 3244 genes, 5 families: tf 1743, gpcr 591, kinase 558, ion 198, immune 154
- HGNC TSV cached (`data/hgnc/hgnc_complete_set.tsv`); Ensembl CDS cached (3244 FASTAs in `data/sequences/`)
- Stratified 70/15/15 split frozen in `data/splits.json` (2270 / 487 / 487)
- Code: `src/data_loader/`, `src/splits/`, `scripts/prepare_data.py`, `scripts/make_splits.py`

## Phase 2 — Encoder embeddings   ✅ done   (deck Week 2)

- DNABERT-2:  3244 × 768  →  `data/embeddings/`, `data/dataset.parquet`
- NT-v2 100M: 3244 × 512  →  `data/embeddings_nt_v2/`, `data/dataset_nt_v2.parquet`

NT-v2 was added beyond the original deck plan as a second encoder so we have converging-evidence on any ceiling we hit.

Code: `src/data_loader/{encoder_runner,nt_v2_encoder}.py`, `scripts/run_encoder.py`, `scripts/run_nt_v2_encoder.py`.

## Phase 3 — Probe + baseline + sanity   ✅ done   (deck Week 3)

| Run | Encoder | Test cos | Test R² | Best config |
|---|---|---:|---:|---|
| Linear probe | DNABERT-2 | 0.9313 | 0.181 | α=10 |
| Linear probe | NT-v2 | 0.9324 | 0.193 | α=10 |
| 4-mer baseline | — | 0.9306 | 0.174 | α=0.01 |
| Anti-baseline | DNABERT-2 | 0.9128 | −0.003 | α=1000 |
| Anti-baseline | NT-v2 | 0.9130 | −0.000 | α=1000 |
| MLP probe | DNABERT-2 | 0.9300 | 0.162 | hidden=(256,), α=0.01 |
| MLP probe | NT-v2 | 0.9325 | 0.189 | hidden=(1024,), α=0.01 |

Outcome: **informative negative** per `framework.md` § Success / failure criteria — both encoders tie the 4-mer baseline within ±0.002 cosine. Anti-baseline R² ≈ 0 confirms the pipeline is honest. MLP depth doesn't move the number, so the ceiling is the representation, not the probe's capacity. See `findings.md` for the full read.

Code: `src/linear_trainer/`, `src/kmer_baseline/`, `scripts/train_{probe,baseline,anti_baseline,mlp_probe}.py`.

## Phase 4 — Qualitative deliverables   ⏳ open   (deck Weeks 4–5)

These are the deck's Week 4–5 commitments that have not yet shipped.

- [ ] **Retrieval@k metric** (k = 1, 5, 10) on linear-probe test predictions for both encoders.
- [ ] **Family-classification accuracy** — logistic regression on `y_hat → family`, compared to the same classifier on real `y`. Measures how much class-level structure survives the projection.
- [ ] **Zero-shot demo** — pick 3–5 uncharacterised genes (poorly annotated or with very short summaries), embed → project → k-NN → predicted family + neighbour symbols + cosines. Markdown table for the write-up.
- [ ] **Visualisation** — PCA + UMAP for DNA space (`x`), text space (`y`), projected space (`y_hat`), colour-coded by family. Six panels total. Save to `data/figures/`.
- [ ] **Captum Integrated Gradients** (Hayden's slot in the deck) — one figure per family, attributions over the CDS for 1–2 representative genes, scalar target = cosine to the family centroid in GenePT space. Look for motif enrichment in high-attribution windows.
- [ ] **Write-up** — intro (from `project.md`) → methods (point at `framework.md` and `src/data_loader/pipeline.md`) → results table + figures from above → discussion in informative-negative framing.

## Phase 5 — Optional ceiling-breaker experiments   🔬 open, lower priority

Strictly after Phase 4. These address the three caveats in `findings.md`.

- [ ] **Pooling sweep** — CLS / max-pool / attention-weighted variants on both encoders. The mean-pool ceiling could be smearing out per-position signal.
- [ ] **Window sweep** — full transcript (promoter + UTR + CDS) instead of CDS-only. CDS is the most composition-homogenous part of a gene because of codon usage; promoters and UTRs may carry more function-discriminating signal.
- [ ] **Optional third encoder** — HyenaDNA, Caduceus, or GENA-LM. Not required: convergence is already cross-encoder with two. Each additional encoder hitting the same ceiling further strengthens the read.

## Resolved / archived

- **Mean-of-chunks pooling vs CLS / max-pool?** → Unresolved empirically; deferred to Phase 5. Cosine plateaued at the 4-mer baseline on both encoders, which was the trigger condition.
- **Five families enough resolution?** → Yes for the current corpus and probe. Sub-family classification is a Phase 4-or-later ask after we see the cluster figures.
- **Version `dataset.parquet` if pooling changes?** → Handled implicitly by adding `dataset_nt_v2.parquet` as a sibling artefact rather than mutating in place. Future encoder/pooling variants follow the same convention.

## Dependency graph

```
[1 data] → [2 encoders] → [3 probe + baseline + sanity] → [4 deliverables] → [5 ceiling-breakers]
                                                                  └→ write-up
```
