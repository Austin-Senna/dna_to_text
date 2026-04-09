# Cross-Modal Alignment: Mapping DNA Latent Space to Gene Ontology

## Overview

Can a pre-trained DNA language model already "understand" gene function without ever being taught it? This project finds out by training a simple linear probe to map **DNABERT-2** sequence embeddings into the text embedding space of **GenePT** (NCBI gene summaries embedded via OpenAI). If a lightweight linear transformation can align these two modalities, it means the DNA model has implicitly learned biological semantics from raw nucleotide syntax alone.

## Core Idea

We take two independently trained embedding spaces — one from DNA sequences, one from natural-language gene descriptions — and ask whether they can be linearly aligned. Success means DNA LLMs encode human-readable function; failure is an equally informative negative result pointing to the limits of current genomic foundation models.

## Data

- **Genes:** 500–1,000 well-characterized human genes spanning 4–5 functional families (kinases, transcription factors, ion channels, immune receptors)
- **DNA embeddings (X):** DNABERT-2 (`zhihan1996/DNABERT-2-117M`) via Hugging Face
- **Text embeddings (Y):** GenePT pre-computed embeddings (NCBI gene summaries)

## Method

1. **Embed** — Extract dense vectors from both modalities for every gene in the dataset.
2. **Probe** — Train a Ridge Regression model (scikit-learn) to project X → Y.
3. **Benchmark** — Compare against a k-mer frequency baseline (identical Ridge Regression on 4-mer counts) using cosine similarity and R² on a held-out test set.
4. **Zero-shot demo** — Pass in poorly characterized / hypothetical gene sequences and infer their functional family from the projected text embedding.
5. **Visualize** — PCA/t-SNE plots of the original DNA space, target text space, and projected space, color-coded by functional family.

## Stack

Python · Hugging Face Transformers · scikit-learn · pandas · UMAP-learn / t-SNE · Captum (Integrated Gradients)

Runs on standard GPUs (DNABERT-2 is only 117M parameters).

## Team

| Member | Focus |
|--------|-------|
| Austin | Data pipeline, embedding generation, evaluation metrics |
| Andrew | Linear probe training, k-mer baseline |
| Hayden | Interpretability (Integrated Gradients), visualization, write-up |