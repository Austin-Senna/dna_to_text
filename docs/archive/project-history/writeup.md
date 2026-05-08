# Writeup — direction document

This is the **structure and content guide** for the project writeup, formatted as a *Bioinformatics* Application Note (≤2 pages main text + supplementary). It is not the final prose — fill each section against this scaffold using the numbers in `docs/findings/findings.md`, `data/full_table.md`, `data/kappa_summary.md`, and the figures in `analysis/viz/figures/` and `analysis/demo/`.

**Audience:** a stranger with a general understanding of the field but no knowledge of this project. Self-contained.

**Submission:** compressed folder uploaded to Courseworks containing the writeup PDF, all source code/scripts, sample input/output files, list of large public files, and a `README` (covered in the deliverables section below).

---

## Bioinformatics Application Note format (target structure)

Author guidelines: <https://academic.oup.com/bioinformatics/pages/author-guidelines>. Application Notes are short — main text ≤2 pages including all display items, plus optional supplementary. Use these section headings.

### Title
One descriptive sentence. Working title:
> *Linear probing of pretrained DNA encoders for cross-modal alignment to gene function*

Keep under ~100 characters. Avoid the word "novel."

### Authors and affiliations
Austin Senna · Andrew · Hayden · Columbia University, Department of Computer Science · contact email.

### Abstract (≤150 words; four labelled paragraphs)

- **Motivation.** One paragraph (~50 words). State the gap: pretrained DNA language models (DNABERT-2, NT-v2) claim to learn general genomic representations from raw sequence, but it is unclear whether their frozen embeddings already carry function-level semantics that a *linear* map can recover. Cross-modal alignment to gene-summary text embeddings (GenePT) provides a direct test.
- **Results.** One paragraph (~50 words). Headline numbers in this order: (1) 5-way family classification — best macro-F1 0.828 / κ 0.821 (NT-v2 + meanD pooling) vs 4-mer baseline 0.672 / 0.702, Δκ +0.119; (2) DNA→text regression — best R² 0.210 (DNABERT-2) vs 4-mer 0.174, ΔR² +0.036; (3) the two encoders' rankings flip across tasks, consistent with their different pretraining recipes (BPE vs 6-mer tokenizer · 135 vs 850 genomes · 512 vs ~6000-token context).
- **Availability and implementation.** One short paragraph. GitHub URL, license (MIT), language (Python ≥3.10), key dependencies (PyTorch, transformers, scikit-learn).
- **Contact.** Corresponding author email.
- **Supplementary information.** "Supplementary data are available at *Bioinformatics* online" (or, for this submission, in the compressed folder).

### 1 Introduction (~250 words)

Three short paragraphs. No deep literature review — this is an Application Note.

- **Paragraph 1 — what's already established.** Pretrained DNA encoders learn from raw sequence via masked-LM. They produce dense embeddings claimed to capture biological structure. Cite DNABERT-2 (Zhou et al. 2024) and Nucleotide Transformer v2 (Dalla-Torre et al. 2024).
- **Paragraph 2 — the gap.** Whether frozen embeddings already encode function-level semantics in a *linearly-readable* form has not been systematically tested across tasks. Linear probing is the standard tool for this question; existing benchmarks focus on tagging tasks and rarely cross modalities.
- **Paragraph 3 — what this work does.** We linearly probe two frozen DNA encoders against (i) a 5-way gene-family classification target and (ii) a 1536-d gene-summary text-embedding target (GenePT, Chen & Zou 2024). We anchor every comparison to a 4-mer composition baseline and a shuffled-label anti-baseline. The two encoders share the masked-LM objective but differ on tokenizer, hidden dim, context window, and pretraining corpus — letting us read out which task each recipe favours.

### 2 Approach / Methods (~400 words, condensed)

The detail is in the GitHub repo; this section gives the reader enough to evaluate the claims. Keep ≤4 short paragraphs.

- **Dataset.** 3244 human protein-coding genes from 5 functional families (TF 1743 · GPCR 591 · kinase 558 · ion channel 198 · immune receptor 154), drawn from HGNC ∩ GenePT. Stratified 70/15/15 train/val/test split on family, seed 42, frozen in `data/splits.json`. Reference Table 1 (family counts).
- **Encoders.** DNABERT-2 117M (BPE tokenizer, 768-d, 512-token context) and NT-v2 multi-species 100M (fixed 6-mer non-overlapping tokenizer, 512-d, ~6000-token context). Both frozen. Each canonical CDS is chunked, encoded, and reduced to a single dense vector via across-chunk mean (the `meanmean` baseline pooling) or one of four alternatives (`meanD` = concat[first, last, mean]; `meanG` = D + max-across-chunks; `maxmean`; `clsmean`). See `src/data_loader/multi_pool.py` and `src/data_loader/pooling_aggregator.py`.
- **Probes.** Ridge regression for the 1536-d GenePT target (metric: macro R²); logistic regression for classification (metrics: macro-F1, Cohen's κ). Regularisation swept on val, refit on train+val, evaluated **once** on test. Source: `src/linear_trainer/`.
- **Baselines and controls.** (i) 4-mer composition: 256-d L1-normalised k-mer histogram, same probe recipe — encoder-independent floor. (ii) Length-only: log(CDS length), 1 feature — tests trivial size cues. (iii) Shuffled-label anti-baseline: labels permuted in train+val only — tests pipeline honesty. Anti-baseline κ lands within ±0.10 of zero on every classification task.

### 3 Results (~600 words, organised around two display items)

This is the longest section. Two figures + one table is plenty.

- **Section 3.1 — Family classification.** Lead with the strong result. Reference **Table 2** (5-way classification, best variant per encoder + 4-mer + length + shuffled, columns: macro-F1, κ, Δκ vs 4-mer). Numbers in `data/full_table.md`. State: NT-v2 + meanD reaches macro-F1 0.828 / κ 0.821, Δκ +0.119 over k-mer; DNABERT-2 best is +0.020. Anti-baseline κ +0.048 — the headroom is real. **Figure 1**: UMAP of NT-v2 meanD embeddings coloured by family, showing five clean clusters (`analysis/viz/figures/umap_nt_v2_meanD.png`).
- **Section 3.2 — Cross-modal alignment to text.** Honest result. Best DNABERT-2 R² 0.210 vs 4-mer 0.174 (ΔR² +0.036); NT-v2 0.193. Note the encoder ranking *flips* between tasks. Frame this as evidence that pretraining recipe shapes which downstream task each encoder wins — not as a contradiction. R² of 0.21 is modest and bounded by target noise; this is a partial, not decisive, recovery.
- **Section 3.3 — Recipe-shape-readout.** Brief paragraph connecting the per-task encoder rankings back to the recipe table from Section 2. NT-v2's broader pretraining (850 species) and longer context favour global family signal; DNABERT-2's BPE may favour finer-grained motif units that align with text. **Figure 2** (optional, supplementary if space): two-panel UMAP showing DNABERT-2 before / after the chunk-boundary tokenisation fix (`analysis/viz/figures/umap_dnabert2_tokenisation_compare.png`) — concrete illustration that recipe details (here, whether CLS was pretrained as a sequence summary) gate what a linear probe can read out.
- **Section 3.4 — Zero-shot demo.** One sentence. Four held-out / sparsely-annotated genes (e.g., ZNF839, ZNHIT2) classified correctly via the family-classifier head. Reference `analysis/demo/output.md` for full predictions.

### 4 Discussion (~150 words)

Three short points.
1. **What's supported.** A linear probe over a frozen DNA encoder recovers gene family identity well beyond k-mer composition (Δκ +0.12). For cross-modal regression to text, the gain over k-mer is real but small (ΔR² +0.04) — alignment is partial, not decisive.
2. **What's not.** No single encoder dominates. Whether tokenizer, corpus, or context window is the responsible axis cannot be isolated without a controlled study (matched-recipe pairs varying one axis at a time).
3. **Limitations.** One organism (human), CDS-only (no UTR / promoter), two encoders, ~3.2k genes. A third encoder (HyenaDNA / Caduceus / GENA-LM) would test whether the family-recovery claim is architecture-general. Larger / longer-context encoders would test whether the regression ceiling moves.

### Acknowledgements
Course staff; any compute provider.

### Funding
None / N/A.

### References
Use the *Bioinformatics* numeric or author-year style. At minimum:
- Zhou *et al.* 2024 — DNABERT-2.
- Dalla-Torre *et al.* 2024 — Nucleotide Transformer v2.
- Chen & Zou 2024 — GenePT.
- Pedregosa *et al.* 2011 — scikit-learn.
- McInnes *et al.* 2018 — UMAP.

---

## Display items

**Tables (in main text):**
- **Table 1.** Dataset composition by family (count, HGNC selection rule). Source: `docs/project-history/project.md` table.
- **Table 2.** Headline results — best variant per encoder vs baselines, three classification tasks + regression in one matrix; columns include macro-F1, κ, R², and Δ vs 4-mer. Condense from `data/full_table.md`.

**Figures (in main text):**
- **Figure 1.** UMAP of NT-v2 meanD embeddings, coloured by family. File: `analysis/viz/figures/umap_nt_v2_meanD.png`.
- **Figure 2.** Two-panel before/after tokenisation UMAP for DNABERT-2 (or move to supplementary). File: `analysis/viz/figures/umap_dnabert2_tokenisation_compare.png`.

**Supplementary (in compressed folder):**
- **Supp. Table S1.** Full results matrix — every (encoder × pooling × task) cell, both metrics. Use `data/full_table.md`.
- **Supp. Table S2.** Cohen's κ summary per task. Use `data/kappa_summary.md`.
- **Supp. Note S1.** Pooling sweep details and the chunk-boundary tokenisation finding. Source: `docs/findings/findings.md` Phase 4b.
- **Supp. Note S2.** Zero-shot demo predictions on poorly-characterised genes. Source: `analysis/demo/output.md`.

---

## Submission deliverables (per assignment)

The compressed folder uploaded to Courseworks must contain:

1. **The writeup PDF.** Compiled from this scaffold.
2. **Pointer to the GitHub repo.** Add the URL prominently in the abstract's "Availability and implementation" line and again in the README.
3. **All source code / scripts.** Already in the repo. Includes:
   - `src/` — data loader, pooling, probes, baselines.
   - `scripts/` — runners for each phase, including `build_full_table.py`, `compute_kappa.py`, `train_logistic_probe.py`, `train_ridge_probe.py`.
   - `analysis/viz/` — code that generates Figures 1–2.
   - `analysis/demo/zero_shot.py` — code for the zero-shot demo.
4. **Sample small input and output files.** Make sure the repo includes:
   - A small input sample (e.g., a 10-gene parquet slice or the first N rows of `data/dataset.parquet`).
   - The corresponding output (small `metrics.json` slice, a sample confusion matrix, the demo `output.md`).
   - If these don't yet exist as small standalone files, add `data/sample_input.parquet` and `data/sample_output.json` (or equivalent) with a few rows so a reviewer can run the pipeline end-to-end without the full corpus.
5. **List of large public files.** In the README, name the large files the project operates on and where to get them:
   - `data/dataset.parquet` — built from HGNC complete gene set + Ensembl CDS fetches + GenePT embedding table. Provide the build command (likely `scripts/build_dataset.py` or similar) and the public sources.
   - GenePT embedding table — Chen & Zou 2024 release URL.
   - DNABERT-2 / NT-v2 model weights — Hugging Face repo IDs (`zhihan1996/DNABERT-2-117M`, `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`).
6. **README file.** Must specify:
   - **What each top-level file/dir is.** One-line description per item.
   - **System requirements.** Python ≥3.10; PyTorch ≥2.0; CUDA optional; Apple Silicon MPS supported. Single consumer GPU sufficient (DNABERT-2 117M and NT-v2 100M both fit).
   - **Dependencies.** Reference `requirements.txt` or `pyproject.toml`. Include exact versions for `torch`, `transformers`, `scikit-learn`, `pandas`, `umap-learn`.
   - **How to test-run on sample inputs.** Concrete command sequence: `pip install -r requirements.txt` → `python scripts/build_dataset.py --sample` (or pre-shipped sample) → `python scripts/train_logistic_probe.py --encoder nt_v2 --task family5 --sample` → expected output (specific metric values within tolerance).
   - **How to run on the full data.** Replace `--sample` with full-data flags; note expected wall-clock per phase.
   - **Parameters / flags.** Document every CLI argument used in the writeup's reproducibility line.

---

## Working order

Suggested fill-in order (highest to lowest leverage):

1. Lock the abstract numbers — those gate every other section.
2. Methods (Section 2) — write to half the target length, then trim.
3. Results (Section 3.1, 3.2) — these carry the paper. Get Table 2 + Figure 1 typeset before writing prose around them.
4. Introduction (Section 1) — easier once results are concrete.
5. Discussion (Section 4) — keep it short; resist scope creep.
6. README + sample I/O files — required for submission, easy to underestimate.
7. Final pass: check ≤2 pages including display items, all references resolve, GitHub link works, sample run reproduces a sample number.
