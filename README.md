# dna_to_text

Cross-modal alignment between DNABERT-2 and GenePT embedding spaces. See `project.md` for the research idea.

## Setup

```bash
uv venv
uv pip install -e .
```

GenePT artifacts live in `GenePT_emebdding_v2/` (gitignored):
- `GenePT_gene_embedding_ada_text.pickle`
- `NCBI_summary_of_genes.json`

## Pipeline

Two stages — CPU-only data prep, then DNABERT-2 encoding.

```bash
# 1. inspect what's in GenePT
python scripts/prepare_data.py --analyze-only

# 2. CPU: build gene table + fetch Ensembl CDS
python scripts/prepare_data.py --limit 5     # smoke test
python scripts/prepare_data.py               # full run

# 3. GPU/MPS/CPU: embed CDS with DNABERT-2 (117M params, runs locally)
python scripts/run_encoder.py --device mps   # or cuda / cpu
```

Final output: `data/dataset.parquet` with `symbol, ensembl_id, family, summary, x (DNABERT-2 768d), y (GenePT 1536d)`.

All caches (`data/sequences/`, `data/embeddings/`, `data/hgnc/`) are reused on rerun.
