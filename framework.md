# Framework

How the experiment is structured. Inputs are `data/dataset.parquet` (DNABERT-2 X) and `data/dataset_nt_v2.parquet` (NT-v2 X), each with columns `symbol, ensembl_id, family, summary, x, y`. Data preparation lives upstream in `src/data_loader/pipeline.md`.

## Hypothesis

A linear map `W: R^d_in -> R^1536` can transport DNA-language-model CDS embeddings into GenePT text-embedding space well enough to recover gene function. Tested on two encoders (`d_in ∈ {768, 512}`) against a 4-mer composition baseline and a shuffled-Y anti-baseline.

## Dataset

3244 human protein-coding genes spanning 5 functional families. Selection mechanism: regex match on HGNC's `gene_group` text column → drop rows without `ensembl_id` → intersect with the GenePT symbol set → first-family-wins dedup so cross-family duplicates are removed.

| Family | Includes (`gene_group` regex, simplified) | Excludes | Count |
|---|---|---|---:|
| `tf` | zinc finger, homeobox, bHLH, bZIP, forkhead, HMG, nuclear receptor, T-box, SOX, ETS, "transcription factor" | binding, cofactor | 1743 |
| `gpcr` | "G protein-coupled receptor", adrenoceptor, 5-HT, dopamine, muscarinic, opioid, chemokine, olfactory | — | 591 |
| `kinase` | `\bkinase` | inhibitor, regulator, substrate, pseudokinase | 558 |
| `ion` | `\bchannel` | regulat, auxiliary, interacting | 198 |
| `immune` | TLR, IL receptor, Fc receptor, NLR, KIR, TCR, BCR, C-type lectin, Ig-like receptor | binding | 154 |
| | | **Total** | **3244** |

Parquet schema:

| Column | Type | Notes |
|---|---|---|
| `symbol` | str | HGNC gene symbol (matches GenePT key) |
| `ensembl_id` | str | Ensembl gene ID (no version suffix) |
| `family` | str | one of `tf, gpcr, kinase, ion, immune` |
| `summary` | str | NCBI gene summary text |
| `y` | float32[1536] | GenePT (OpenAI ada-002) embedding of `summary` |
| `x` | float32[768] | DNABERT-2 mean-pooled CDS embedding (in `dataset.parquet`) |
| `x` | float32[512] | NT-v2 mean-pooled CDS embedding (in `dataset_nt_v2.parquet`) |

See `src/data_loader/pipeline.md` for the upstream pipeline (HGNC join, Ensembl REST CDS fetch, encoder runs, caching).

## Splits

Stratified 70/15/15 by `family`, seed 42. Frozen in `data/splits.json`. Train 2270 / val 487 / test 487. Loaded via `splits.load_split(name, dataset_path=...)`; the shuffled-Y control uses `splits.load_shuffled_y(name, seed=...)`.

## Encoders

| Encoder | Params | Tokeniser | Dim | Chunk × stride | Pretraining |
|---|---|---|---|---|---|
| `zhihan1996/DNABERT-2-117M` | 117M | BPE | 768 | 512 × 64 | 135 genomes (multi-species) |
| `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | 100M | 6-mer non-overlapping | 512 | 1000 × 64 | 850 genomes (multi-species) |

Both encoders are run on the canonical CDS (one transcript per gene, fetched via Ensembl REST), tokenised without special tokens, windowed into chunks, mean-pooled per chunk over the token dimension, then mean across chunks. Implementation: `src/data_loader/encoder_runner.py` (DNABERT-2) and `src/data_loader/nt_v2_encoder.py` (NT-v2).

## Probes

### Linear probe (primary)

Ridge Regression, multi-output: `y_hat = X W + b`. `sklearn.linear_model.Ridge`, alpha swept on val over `[1e-2, 1e-1, 1, 10, 100, 1000]`, refit on train+val at the chosen alpha, evaluated once on test. No nonlinearity by design — the whole point is to probe linear alignment. Implementation: `src/linear_trainer/probe.py`.

### 4-mer baseline (control)

Same Ridge recipe on 256-d L1-normalised 4-mer frequency vectors of the CDS. Encoder-independent — the baseline number is identical regardless of which X we use. Tells us how much of the signal comes from a learned representation vs raw nucleotide composition. Implementation: `src/kmer_baseline/featurizer.py`.

### Shuffled-Y anti-baseline (sanity gate)

Same Ridge recipe with `Y` permuted (seed 42) in train+val, then evaluated against real test `Y`. Catches pipeline leakage. Should collapse to R² ≈ 0; cosine should fall to GenePT's anisotropy floor (~0.91, the cosine of mean-Y to any real GenePT vector). If it doesn't, the pipeline is leaking and the real probe's numbers are suspect.

### MLP probe (diagnostic)

`sklearn.neural_network.MLPRegressor`, 1-hidden-layer ReLU, sweep `hidden ∈ {(256,), (512,), (1024,)}` × `alpha ∈ {1e-4, 1e-3, 1e-2}` with early-stopping. Same train/val/refit/test protocol. Diagnostic only: if it meaningfully clears the linear probe's R², the encoder has nonlinear structure worth chasing with a bigger model or fine-tune. Implementation: `src/linear_trainer/mlp_probe.py`.

## Metrics

Computed on the held-out test split.

| Metric | Definition | Status |
|---|---|---|
| Cosine (mean, median) | `cos(y_hat_i, y_i)` per gene | Implemented in every trainer script |
| Macro R² | `r2_score(Y_te, Y_hat, multioutput="uniform_average")` over 1536 dims | Implemented in every trainer script |
| Retrieval@k | For each test `y_hat_i`, rank all real summaries; is `summary_i` in top-k? | Planned (Phase 4) |
| Family classification accuracy | Logistic regression on `y_hat → family`; compare to same classifier on real `y` | Planned (Phase 4) |

Cosine is misleading as a headline because of GenePT's anisotropy floor (mean-Y has cosine ≈ 0.91 with any real GenePT vector). R² is the honest signal — see `findings.md`.

## Zero-shot demo (planned, Phase 4)

For each gene in a small uncharacterised hold-out set:

1. Fetch CDS, embed with the chosen encoder → `x`.
2. Project: `y_hat = x W + b`.
3. k-NN against real GenePT space → predicted functional family (majority vote of top-k).
4. Report predicted family, top-5 neighbour genes, cosine to centroid of each family.

Success criterion is qualitative: the predicted family should be biologically plausible given any available annotation.

## Visualisation (planned, Phase 4)

Three 2-D plots, same points, coloured by `family`:

1. DNA space (`x`) — PCA and UMAP.
2. Text space (`y`) — PCA and UMAP.
3. Projected space (`y_hat`) — PCA and UMAP on test split.

Useful if (2) and (3) show the same cluster topology; damning if (1) and (3) do.

## Interpretability (planned, Phase 4)

Captum Integrated Gradients with the encoder + frozen probe stacked, scalar target = cosine to a chosen family centroid in GenePT space. Aggregate token-level attributions back to nucleotide windows; look for motif enrichment inside family-specific attributed regions (kinase domains, zinc-finger motifs, TM helices for GPCRs, etc.).

## Success / failure criteria

- **Positive result:** probe beats 4-mer baseline on cosine and retrieval@k, and family clusters are visibly preserved in projected space.
- **Informative negative:** probe matches or barely beats 4-mer baseline → DNA encoder's extra capacity is not functional, it is compositional. Still publishable as a limits-of-genomic-LLMs finding. **This is the empirically observed outcome — see `findings.md`.**
- **Pipeline bug:** anti-baseline (shuffled `y`) scores non-trivially. Stop and debug before drawing any conclusion.

## Repo layout

```
src/
  data_loader/
    dataset_loader.py     # GenePT pickle + HGNC TSV + family regex selection
    sequence_fetcher.py   # Ensembl REST → cached CDS FASTAs
    encoder_runner.py     # DNABERT-2 chunk + mean-pool
    nt_v2_encoder.py      # NT-v2 chunk + mean-pool
    pipeline.md           # data-prep README
  splits/
    make_splits.py        # stratified 70/15/15 builder
    loader.py             # load_split, load_shuffled_y
  linear_trainer/
    probe.py              # Ridge probe + alpha sweep
    mlp_probe.py          # diagnostic MLPRegressor probe
  kmer_baseline/
    featurizer.py         # 4-mer L1-normalised featuriser + load_kmer_features

scripts/
  prepare_data.py            # CPU: GenePT + HGNC + Ensembl → gene_table.parquet
  run_encoder.py             # GPU/MPS: DNABERT-2 → dataset.parquet
  run_nt_v2_encoder.py       # GPU/MPS: NT-v2 → dataset_nt_v2.parquet
  make_splits.py             # CLI for splits builder
  train_probe.py             # linear probe (works on either dataset via --dataset)
  train_baseline.py          # 4-mer baseline
  train_anti_baseline.py     # shuffled-Y sanity gate
  train_mlp_probe.py         # diagnostic MLP probe
  inspect_data.py            # human-readable dump of every artefact
  inspect_families.py        # preview family selection before full run

data/
  hgnc/hgnc_complete_set.tsv
  sequences/{ENSG…}.fa             # cached canonical CDS, one file per gene
  embeddings/{ENSG…}.npy           # DNABERT-2 768-d cache
  embeddings_nt_v2/{ENSG…}.npy     # NT-v2 512-d cache
  gene_table.parquet               # post-prepare table (no x yet)
  dataset.parquet                  # final: includes DNABERT-2 x + GenePT y
  dataset_nt_v2.parquet            # final: includes NT-v2 x + GenePT y
  splits.json                      # frozen 70/15/15 ensembl_ids
  probe.npz                        # trained DNABERT-2 linear probe (W, b, alpha)
  probe_nt_v2.npz                  # trained NT-v2 linear probe
  metrics.json                     # all runs, all metrics, append-only
```

## References

**Used in this project**

- Zhou et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* 2024. Model: `zhihan1996/DNABERT-2-117M`.
- Dalla-Torre et al. *The Nucleotide Transformer: building and evaluating robust foundation models for human genomics.* 2024 (InstaDeepAI). Model: `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`.
- Chen, Zou. *GenePT: a simple but effective foundation model for genes and cells built from ChatGPT.* 2024. Source of the 1536-d text targets.
- HGNC database — https://www.genenames.org (HUGO Gene Nomenclature Committee). Source of gene symbol → group → Ensembl ID join.
- Ensembl REST API — https://rest.ensembl.org. Source of canonical CDS sequences.
- Pedregosa et al. *scikit-learn: Machine Learning in Python.* JMLR 2011. `Ridge`, `MLPRegressor`, `train_test_split`.
- Wolf et al. *Transformers: State-of-the-art Natural Language Processing.* EMNLP 2020. Loader for both DNA encoders.
- Kokhlikyan et al. *Captum: A unified and generic model interpretability library for PyTorch.* 2020. Planned for Phase 4 (IG attribution).

**Framing only (cited in pitch deck, not used in experiments)**

- Fan et al. *Omni-DNA.* 2025. Generative DNA→text baseline cited for related-work context.
- Li et al. *Alignment or Integration?* 2026. SeqCLIP-style framing.
- Benegas et al. *DNAChunker.* 2024. Mentioned in the original pitch deck; not used in this repo.
