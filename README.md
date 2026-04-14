# dna_to_text

Cross-modal alignment between DNABERT-2 and GenePT embedding spaces. See `project.md` for the research idea.

## Setup

```bash
uv venv
uv pip install -e .
```

Use `uv run ...` for commands unless you have already activated `.venv/`. In this repo, bare `pip` may not exist on your shell path even though the virtualenv is present.

GenePT artifacts live in `GenePT_emebdding_v2/` (gitignored):
- `GenePT_gene_embedding_ada_text.pickle`
- `NCBI_summary_of_genes.json`

## Pipeline

Two stages — CPU-only data prep, then DNABERT-2 encoding.

```bash
# 1. inspect what's in GenePT
uv run python scripts/prepare_data.py --analyze-only

# 2. CPU: build gene table + fetch Ensembl CDS
uv run python scripts/prepare_data.py --limit 5     # smoke test
uv run python scripts/prepare_data.py               # full run

# 3. GPU/MPS/CPU: embed CDS with DNABERT-2 (117M params, runs locally)
uv run python scripts/run_encoder.py --device auto  # auto -> cuda, then mps, then cpu
```

Final output: `data/dataset.parquet` with `symbol, ensembl_id, family, summary, x (DNABERT-2 768d), y (GenePT 1536d)`.

All caches (`data/sequences/`, `data/embeddings/`, `data/hgnc/`) are reused on rerun.

## Troubleshooting

- `pip: command not found`: use `uv run python ...` to run scripts and `uv pip ...` to manage packages inside the project environment.
- `uv pip install triton` fails on Apple Silicon + Python 3.12: this is expected here. Triton wheels are not available for this platform combination.
- DNABERT-2 on this repo does not actually require Triton for inference. The remote model code is supposed to fall back to plain PyTorch attention when Triton is missing.
- The runtime failure came from a `transformers` remote-code import check that treated DNABERT-2's optional `flash_attn_triton` module as mandatory. The loader now works around that and pins the model revision used by the repo.
- Do not try to speed this up by launching multiple concurrent encoder runs on the same `mps` or GPU device. Keep one encoder process per device; parallel fan-out is not the intended knob for this pipeline.
