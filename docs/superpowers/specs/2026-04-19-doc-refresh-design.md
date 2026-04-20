# Doc Refresh — Design

Refresh the four top-level project docs so they match the current state of `data/` and `src/`. Scope: `project.md`, `framework.md`, `findings.md`, `next_steps.md`.

Reference inputs:
- Current docs: `project.md`, `framework.md`, `findings.md`, `next_steps.md`
- Original pitch deck: `Genomics Project (1).pdf` (slides 2–7 frame the project; slide 8 is the Week 1–5 timeline used to scaffold `next_steps.md`)
- Repo: `src/{data_loader,splits,linear_trainer,kmer_baseline}/`, `scripts/`, `data/`, `src/data_loader/pipeline.md`

## Why

The four docs drifted off the repo:

| Doc | Drift |
|---|---|
| `project.md` | Says "500–1,000 genes" (actual 3244), DNABERT-2 only (also have NT-v2), PCA/t-SNE (we use UMAP), no anti-baseline mention. |
| `framework.md` | DNABERT-2-only framing; `Repo layout` block lists files that don't exist (`src/dna_to_text/probe.py`, etc.); no NT-v2; no MLP probe. |
| `findings.md` | Mostly current; lacks an explicit pointer to `framework.md` for methods detail. |
| `next_steps.md` | Still in phase 2/3 terms even though splits, probe, baseline, anti-baseline, MLP, and NT-v2 all shipped. Open questions about pooling are now empirically resolved. |

User decisions (locked in during brainstorming):
- **Voice (Q1):** combine **B + C**. `project.md` is a frozen pitch with refreshed numbers; `framework.md` and `findings.md` reflect current reality; `next_steps.md` is a full project log (done → in progress → open).
- **Forward scope (Q2):** include both deferred deliverables and ceiling-breaker experiments in `next_steps.md`.
- **Phase 5 ordering (Q3):** ceiling-breakers strictly **after** the deck's Week 4–5 deliverables.
- **References (Q4):** include both the original deck citations and every new tool/model actually used (notably NT-v2). Live in `framework.md`, lightly mirrored in `project.md`.

Out of scope: writing the actual content of Phase 4/5 deliverables, changes to `README.md`, changes to `src/data_loader/pipeline.md` (already accurate).

## Source-of-truth numbers

These are the current empirical numbers each doc must match. Pulled from `data/dataset.parquet`, `data/splits.json`, `data/metrics.json`.

**Corpus** — `data/dataset.parquet`, columns `[symbol, ensembl_id, family, summary, y, x]`, 3244 rows.

| Family | Count |
|---|---:|
| tf | 1743 |
| gpcr | 591 |
| kinase | 558 |
| ion | 198 |
| immune | 154 |
| **total** | **3244** |

**Splits** — stratified by `family`, seed 42, `data/splits.json`. Train 2270 / val 487 / test 487.

**Encoders**
- DNABERT-2 (`zhihan1996/DNABERT-2-117M`): 117M params, BPE tokeniser, ALiBi, 768-d. Mean-pool over 512-token chunks with 64-token overlap. Vectors in `data/embeddings/{ENSG…}.npy`. Final table `data/dataset.parquet`.
- NT-v2 100M multi-species (`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`): 100M params, 6-mer non-overlapping tokeniser, rotary embeddings, GLU FFN, 512-d. Mean-pool over 1000-token chunks with 64-token overlap. Vectors in `data/embeddings_nt_v2/{ENSG…}.npy`. Final table `data/dataset_nt_v2.parquet`.

**Results** — from `data/metrics.json` (latest run per model per dataset).

| Model | Encoder | Test mean cos | Test median cos | Test R² macro | Best alpha / config |
|---|---|---:|---:|---:|---|
| Linear probe | DNABERT-2 | 0.9313 | 0.9221 | 0.1812 | α=10 |
| Linear probe | NT-v2 | 0.9324 | 0.9254 | 0.1926 | α=10 |
| 4-mer baseline | — | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| Anti-baseline (DNABERT-2) | DNABERT-2 | 0.9128 | 0.9131 | −0.003 | α=1000 |
| Anti-baseline (NT-v2) | NT-v2 | 0.9130 | 0.9141 | −0.000 | α=1000 |
| MLP probe | DNABERT-2 | 0.9300 | 0.9220 | 0.1616 | hidden=(256,), α=0.01 |
| MLP probe | NT-v2 | 0.9325 | 0.9253 | 0.1888 | hidden=(1024,), α=0.01 |

**Repo layout** (current):
```
src/
  data_loader/      # GenePT loader, Ensembl CDS fetcher, DNABERT-2 + NT-v2 encoders
    pipeline.md     # data-prep README
  splits/           # make_splits, load_split, load_shuffled_y
  linear_trainer/   # Ridge probe + sklearn MLPRegressor probe
  kmer_baseline/    # 4-mer L1-normalised featuriser
scripts/
  prepare_data.py
  run_encoder.py            # DNABERT-2
  run_nt_v2_encoder.py
  make_splits.py
  train_probe.py
  train_baseline.py
  train_anti_baseline.py
  train_mlp_probe.py
  inspect_data.py
  inspect_families.py
data/
  dataset.parquet           # DNABERT-2
  dataset_nt_v2.parquet     # NT-v2
  splits.json
  probe.npz, probe_nt_v2.npz
  metrics.json
  embeddings/, embeddings_nt_v2/, sequences/, hgnc/, gene_table.parquet
```

## Per-doc plan

### `project.md` — frozen pitch, numbers refreshed

Tone: aspirational, no result spoiler. Mirrors slides 2–7 of the deck on one page.

Sections (top to bottom):
1. **Title + tagline** — keep `# Cross-Modal Alignment: Mapping DNA Latent Space to Gene Ontology` and the existing one-paragraph overview, but update the pitch to mention "DNABERT-2 and NT-v2" instead of "DNABERT-2" alone.
2. **Core Idea** — keep as-is.
3. **Dataset** *(new section, replaces existing "Data" bullet)* — list the 5 families with one-line descriptions and counts; one sentence on selection mechanism (HGNC `gene_group` substring match, intersected with GenePT and Ensembl, deduped); cite final corpus size = 3244 genes; one sentence on splits (stratified 70/15/15).
4. **Method** — DNABERT-2 + NT-v2 as two encoders, GenePT as targets, Ridge probe, 4-mer baseline, shuffled-Y anti-baseline. PCA/t-SNE → **PCA/UMAP**.
5. **Stack** — Hugging Face Transformers, scikit-learn, pandas, UMAP-learn, Captum (planned). Drop t-SNE.
6. **Team** — keep as-is.
7. **References** *(new, three lines)* — DNABERT-2, Nucleotide Transformer v2, GenePT.

No mention of the actual finding (informative negative). Document remains a forward-looking pitch.

### `framework.md` — methods, current reality

Full rewrite. Sections:

1. **Hypothesis** — A linear map `W: R^d_in -> R^1536` can transport DNA-language-model CDS embeddings into GenePT space well enough to recover gene function. Tested on two encoders (`d_in ∈ {768, 512}`) against a 4-mer baseline and a shuffled-Y anti-baseline.

2. **Dataset**
   - 3244 human protein-coding genes, 5 families.
   - Family table: short name, regex include/exclude rules (paraphrased from `dataset_loader.py::FAMILIES`), per-family count.
   - Selection mechanism in two sentences: HGNC `gene_group` substring regex → drop without `ensembl_id` → intersect with GenePT symbol set → first-family-wins dedup.
   - Parquet schema: `symbol, ensembl_id, family, summary, y (1536-d GenePT ada-002), x (768-d DNABERT-2 in dataset.parquet | 512-d NT-v2 in dataset_nt_v2.parquet)`.
   - Pipeline cross-reference: `src/data_loader/pipeline.md`.

3. **Splits** — 2270 train / 487 val / 487 test, stratified by `family`, seed 42, frozen in `data/splits.json`. Loaded via `splits.load_split(name, dataset_path=...)`.

4. **Encoders** — DNABERT-2 and NT-v2 with the chunk + pooling spec above. One-paragraph rationale per encoder (BPE vs 6-mer; ALiBi vs rotary; 135 vs 850 pretraining genomes).

5. **Probes**
   - Linear (Ridge) — primary. `sklearn.linear_model.Ridge`, multi-output. Alpha sweep `[1e-2, 1e-1, 1, 10, 100, 1000]` on val, refit on train+val, eval on test.
   - 4-mer baseline — same Ridge recipe on 256-d L1-normalised 4-mer counts of the CDS. Encoder-independent; identical numbers regardless of `x`.
   - Anti-baseline — same Ridge recipe with `Y` permuted in train+val (seed 42), evaluated against real test `Y`. Sanity gate.
   - MLP probe — diagnostic. `sklearn.neural_network.MLPRegressor`, 1-hidden-layer ReLU, sweep over `hidden ∈ {(256,), (512,), (1024,)}` × `alpha ∈ {1e-4, 1e-3, 1e-2}`, early-stopping. Same train/val/refit/test protocol.

6. **Metrics**
   - Cosine (mean, median across test).
   - Macro R² across the 1536 output dims.
   - **Planned, not yet computed:** retrieval@k (rank real summaries by predicted vector; is the gene's own summary in top-k); family-classification accuracy (logistic regression on `y_hat` → `family`, vs. same classifier on real `y`).

7. **Zero-shot demo** *(planned, deferred to Phase 4)* — fetch CDS for an uncharacterised gene → embed → project → k-NN against training `y` → majority-vote family. Same as deck slide 7.

8. **Visualisation** *(planned, deferred to Phase 4)* — three 2-D plots (PCA + UMAP) coloured by family: DNA space (`x`), text space (`y`), projected space (`y_hat` on test).

9. **Interpretability** *(planned, deferred to Phase 4)* — Captum Integrated Gradients with the encoder + frozen probe stacked, scalar target = cosine to a chosen family centroid in GenePT space. Look for motif enrichment in high-attribution windows.

10. **Success / failure criteria** — keep the three from the original (positive / informative-negative / pipeline-bug). Add one sentence noting that the **informative-negative** criterion is the empirically observed outcome (pointer to `findings.md`).

11. **Repo layout** — replace stale `src/dna_to_text/...` block with the actual layout above.

12. **References** — full list (used + framing). See References section below.

### `findings.md` — light edits only

Keep current content. Edits:
- Add a one-line pointer near the top: `> Methods detail: see framework.md.`
- Verify result tables match `data/metrics.json` (they do — checked).
- No structural changes; this doc is already current.

### `next_steps.md` — full project log scaffolded on the deck Week 1–5

Replace the entire file. New structure:

```
# Next Steps

Living project log. Phases 1–3 are done; phase 4 carries the deck's Week 4–5
deliverables; phase 5 collects the open scientific questions surfaced by
findings.md.

## Phase 1 — Data pipeline   ✅ done   (deck Weeks 1–2)
- 3244 genes, 5 families (tf 1743, gpcr 591, kinase 558, ion 198, immune 154)
- HGNC TSV cached; Ensembl CDS cached (3244 FASTAs in data/sequences/)
- Stratified 70/15/15 split frozen in data/splits.json (2270 / 487 / 487)
- Code: src/data_loader/, src/splits/, scripts/prepare_data.py, scripts/make_splits.py

## Phase 2 — Encoder embeddings   ✅ done   (deck Week 2)
- DNABERT-2:  3244 × 768  →  data/embeddings/, data/dataset.parquet
- NT-v2 100M: 3244 × 512  →  data/embeddings_nt_v2/, data/dataset_nt_v2.parquet
  (added beyond the original deck plan as a second encoder for converging-evidence on the ceiling)
- Code: src/data_loader/{encoder_runner,nt_v2_encoder}.py, scripts/run_encoder.py, scripts/run_nt_v2_encoder.py

## Phase 3 — Probe + baseline + sanity   ✅ done   (deck Week 3)
- Linear probe (DNABERT-2):  test cos 0.9313  /  R² 0.181   (α=10)
- Linear probe (NT-v2):      test cos 0.9324  /  R² 0.193   (α=10)
- 4-mer baseline:            test cos 0.9306  /  R² 0.174   (α=0.01)
- Anti-baseline (both):      test cos 0.913   /  R² ≈ 0     (sanity gate passes)
- MLP probe (diagnostic):    ties linear probe within ±0.002 cos on both encoders
- Result: informative-negative outcome from framework.md (encoders ≈ 4-mer baseline)
- See: findings.md
- Code: src/linear_trainer/, src/kmer_baseline/, scripts/train_{probe,baseline,anti_baseline,mlp_probe}.py

## Phase 4 — Qualitative deliverables   ⏳ open   (deck Weeks 4–5)
- [ ] Retrieval@k metric (k = 1, 5, 10) on linear-probe test predictions
- [ ] Family-classification accuracy: logistic regression on y_hat → family,
      compared to the same classifier on real y
- [ ] Zero-shot demo: pick 3–5 uncharacterised genes, embed → project → k-NN,
      report predicted family + neighbour symbols + cosines
- [ ] Visualisation: PCA + UMAP for DNA space (x), text space (y), projected (y_hat),
      colour-coded by family. Six panels total. Save to data/figures/.
- [ ] Captum Integrated Gradients (Hayden's slot in the deck): one figure per family,
      attributions over the CDS for 1–2 representative genes, scalar target =
      cosine to the family centroid in GenePT space
- [ ] Write-up: intro (from project.md) → methods (point at framework.md) →
      results (table + figures from above) → discussion (informative-negative framing)

## Phase 5 — Optional ceiling-breaker experiments   🔬 open, lower priority
(strictly after Phase 4 — these address findings.md's "Caveats" section)
- [ ] Pooling sweep: CLS / max-pool / attention-weighted variants on both encoders
- [ ] Window sweep: full transcript (promoter + UTR + CDS) instead of CDS-only
- [ ] Optional third encoder: HyenaDNA, Caduceus, or GENA-LM
      (not required — convergence is already cross-encoder with two)

## Resolved / archived

- Mean-of-chunks pooling vs CLS / max-pool? → unresolved empirically;
  flagged for Phase 5. Cosine plateaued at baseline as predicted.
- 5 families enough resolution? → yes for the current corpus and probe;
  defer harder sub-family classification until after Phase 4 figures.
- Version dataset.parquet if pooling changes? → handled implicitly by adding
  dataset_nt_v2.parquet as a sibling artefact rather than mutating in place.

## Dependency graph

[1 data] → [2 encoders] → [3 probe + baseline + sanity] → [4 deliverables] → [5 ceiling-breakers]
                                                              └→ write-up
```

## References (used in `framework.md`, lightly mirrored in `project.md`)

**Used**
- Zhou et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* 2024. Model `zhihan1996/DNABERT-2-117M`.
- Dalla-Torre et al. *The Nucleotide Transformer: building and evaluating robust foundation models for human genomics.* 2024 (InstaDeepAI). Model `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`.
- Chen, Zou. *GenePT: a simple but effective foundation model for genes and cells built from ChatGPT.* 2024.
- HGNC database (Tweedie et al., *Nucleic Acids Research*, 2025). Source of gene symbol + group + Ensembl ID join.
- Ensembl REST API (Yates et al., 2024). Source of canonical CDS sequences.
- Pedregosa et al. *scikit-learn: Machine Learning in Python.* JMLR 2011. `Ridge`, `MLPRegressor`, `train_test_split`.
- Wolf et al. *Transformers: State-of-the-art Natural Language Processing.* EMNLP 2020. Loader for both DNA encoders.
- Kokhlikyan et al. *Captum: A unified and generic model interpretability library for PyTorch.* 2020. **Planned** for Phase 4 (IG attribution).

**Framing only (cited in pitch deck, not used in experiments)**
- Fan et al. *Omni-DNA.* 2025. Generative DNA→text baseline mentioned for context.
- Li et al. *Alignment or Integration?* 2026. SeqCLIP-style framing.
- Benegas et al. *DNAChunker.* 2024. Mentioned in deck; not used in this repo.

## Validation checklist (before declaring the doc refresh done)

- [ ] Every per-family count in any doc matches `data/dataset.parquet`'s `family` value_counts (3244 total).
- [ ] Every metric quoted in any doc matches the latest entry in `data/metrics.json` for its model+dataset.
- [ ] No doc mentions `src/dna_to_text/...` paths (those don't exist).
- [ ] No doc mentions t-SNE (we use UMAP).
- [ ] `next_steps.md` has zero in-progress items (current real state: phases 1–3 done, phase 4–5 not started).
- [ ] `framework.md` cross-references `src/data_loader/pipeline.md` for data-prep detail rather than duplicating.
- [ ] `project.md` does not mention the informative-negative result (stays a forward-looking pitch).
- [ ] `findings.md` carries one new line pointing to `framework.md`.

## Commit plan

Single commit, message: `docs: refresh project, framework, next_steps to match repo state`.

Rationale for one commit: the four files cross-reference each other, and splitting would leave intermediate states with broken claims (e.g. `framework.md` updated but `next_steps.md` still says "phase 3 not started").
