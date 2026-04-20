# Cross-Modal Alignment: Mapping DNA Latent Space to Gene Ontology

## Overview

Can a pre-trained DNA language model already "understand" gene function without ever being taught it? This project finds out by training simple linear probes to map DNA sequence embeddings — from **DNABERT-2** and **NT-v2 100M multi-species** — into the text embedding space of **GenePT** (NCBI gene summaries embedded via OpenAI). If a lightweight linear transformation can align these two modalities, it means the DNA model has implicitly learned biological semantics from raw nucleotide syntax alone.

## Core Idea

We take two independently trained DNA embedding spaces and one text embedding space, and ask whether the DNA spaces can be linearly aligned to the text space. Success means DNA LLMs encode human-readable function; failure is an equally informative negative result pointing to the limits of current genomic foundation models.

## Dataset

3244 human protein-coding genes drawn from 5 functional families. The corpus comes from intersecting the HGNC complete gene set with the GenePT gene-symbol set, then keeping only genes that have an Ensembl ID and a CDS we can fetch. Cross-family duplicates are removed (first family wins).

| Family | Selection rule (HGNC `gene_group`) | Count |
|---|---|---:|
| `tf` | zinc finger / homeobox / bHLH / bZIP / forkhead / HMG / nuclear receptor / T-box / SOX / ETS / "transcription factor" | 1743 |
| `gpcr` | "G protein-coupled receptor" + adrenoceptor / 5-HT / dopamine / muscarinic / opioid / chemokine / olfactory | 591 |
| `kinase` | `\bkinase` (minus inhibitors / regulators / substrates / pseudokinases) | 558 |
| `ion` | `\bchannel` (minus regulators / auxiliary / interacting) | 198 |
| `immune` | TLR / IL receptor / Fc receptor / NLR / KIR / TCR / BCR / C-type lectin / Ig-like receptor | 154 |
| | **Total** | **3244** |

Schema (`data/dataset.parquet`): `symbol`, `ensembl_id`, `family`, `summary`, `y` (1536-d GenePT), `x` (768-d DNABERT-2; the sibling `dataset_nt_v2.parquet` has `x` = 512-d NT-v2 instead).

Stratified 70/15/15 split on `family`, seed 42, frozen in `data/splits.json` → 2270 / 487 / 487.

## Method

1. **Embed (X).** For every gene's canonical CDS, extract one dense vector per encoder:
   - **DNABERT-2** (`zhihan1996/DNABERT-2-117M`) — 117M params, BPE tokeniser, 768-d. Mean-pool over 512-token chunks with 64-token overlap.
   - **NT-v2 100M multi-species** (`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`) — 100M params, 6-mer non-overlapping tokeniser, 512-d. Mean-pool over 1000-token chunks with 64-token overlap.
2. **Target (Y).** GenePT pre-computed embeddings of NCBI gene summaries, 1536-d.
3. **Probe.** Train a Ridge Regression model (scikit-learn, multi-output) per encoder to project X → Y. Sweep `alpha` on val, refit on train+val, evaluate on test.
4. **Controls.**
   - **4-mer frequency baseline** — same Ridge recipe on 256-d L1-normalised 4-mer counts of the CDS. Encoder-independent.
   - **Shuffled-Y anti-baseline** — same Ridge recipe but Y permuted in train+val, evaluated against real test Y. Pipeline sanity gate.
5. **Zero-shot demo.** Pass in poorly characterized / hypothetical gene sequences and infer their functional family from the projected text embedding via k-NN.
6. **Visualize.** PCA and UMAP plots of the original DNA space, target text space, and projected space, color-coded by functional family.

## Stack

Python · Hugging Face Transformers · scikit-learn · pandas · UMAP-learn · Captum (Integrated Gradients)

Runs on standard GPUs. DNABERT-2 (117M) and NT-v2 100M both fit on a single consumer GPU or Apple Silicon MPS.

## Team

| Member | Focus |
|--------|-------|
| Austin | Data pipeline, embedding generation, evaluation metrics |
| Andrew | Linear probe training, k-mer baseline |
| Hayden | Interpretability (Integrated Gradients), visualization, write-up |

## References

- Zhou et al. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* 2024.
- Dalla-Torre et al. *The Nucleotide Transformer: building and evaluating robust foundation models for human genomics.* 2024 (InstaDeepAI).
- Chen, Zou. *GenePT: a simple but effective foundation model for genes and cells built from ChatGPT.* 2024.
