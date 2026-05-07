# dna_to_text

Cross-modal probing of frozen DNA sequence representations against gene-family labels and GenePT text embeddings. The main paper path compares DNABERT-2, NT-v2, GENA-LM, and HyenaDNA on a 3244-gene 5-family classification task, with Enformer reported separately as a supervised sequence-to-function comparator.

**Where to start reading:**
- `project.md` — original research idea and corpus.
- `framework.md` — experimental design (probes, baselines, metrics).
- `findings.md` — running results journal (Phase 3 regression, Phase 4 classification, Phase 5 encoder expansion).
- `writeup.md` — presentation-style summary for the deck.
- `next_steps.md` — phase log + what's open.

## Setup

```bash
uv venv
uv pip install -e .       # or `uv sync`
```

Use `uv run ...` for commands unless you have already activated `.venv/`.

GenePT artifacts live in `GenePT_emebdding_v2/` (gitignored):
- `GenePT_gene_embedding_ada_text.pickle`
- `NCBI_summary_of_genes.json`

## Pipeline (high-level)

Three stages: CPU data prep → encoder embedding → probes / baselines / matrix runs.

```bash
# 1. CPU: build gene table + fetch Ensembl CDS (idempotent, caches to data/sequences/)
uv run python scripts/prepare_data.py

# 2. GPU/MPS/CPU: embed CDS with each encoder (single-vector pipeline; Phase 1-3)
uv run python scripts/run_encoder.py --device auto       # DNABERT-2 -> data/dataset.parquet
uv run python scripts/run_nt_v2_encoder.py --device auto # NT-v2     -> data/dataset_nt_v2.parquet

# 3. Re-extract per-chunk reductions (mean / max / cls per chunk)
uv run python scripts/run_multi_pool_extract.py --encoder dnabert2
uv run python scripts/run_multi_pool_extract.py --encoder nt_v2
uv run python scripts/run_multi_pool_extract.py --encoder gena_lm
uv run python scripts/run_multi_pool_extract.py --encoder hyena_dna
uv run python scripts/build_pooling_datasets.py --encoder dnabert2  # 5 variant parquets
uv run python scripts/build_pooling_datasets.py --encoder nt_v2
uv run python scripts/build_pooling_datasets.py --encoder gena_lm
uv run python scripts/build_pooling_datasets.py --encoder hyena_dna

# 4. Enformer supervised comparator features (optional, separate table)
uv pip install enformer-pytorch
uv run python scripts/run_enformer_features.py --device auto

# 5. Splits; binary subsets are legacy/appendix only
uv run python scripts/make_splits.py
uv run python scripts/make_binary_subsets.py

# 6. Main probes + baselines (family5 is the paper headline task)
uv run python scripts/train_probe.py --dataset data/dataset_nt_v2_meanD.parquet      # Ridge into GenePT
uv run python scripts/train_logistic_probe.py --dataset nt_v2_meanD --task family5    # Logistic 5-way
uv run python scripts/train_logistic_probe.py --dataset gena_lm_meanD --task family5
uv run python scripts/train_logistic_probe.py --dataset hyena_dna_meanG --task family5
uv run python scripts/train_logistic_probe.py --dataset enformer_tracks_center --task family5
uv run python scripts/build_family5_table.py
uv run python scripts/build_regression_table.py

# 7. Demo + visualisations for the deck
uv run python demo/zero_shot.py                  # demo/output.md
uv run python viz/umap_meanD.py                  # viz/figures/umap_nt_v2_meanD.png
uv run python viz/umap_tokenisation_compare.py   # viz/figures/umap_dnabert2_tokenisation_compare.png

# 8. Optional TSS-context self-supervised encoder ablation
uv run python scripts/run_tss_multi_pool_extract.py --encoder nt_v2 --device auto
uv run python scripts/build_tss_pooling_datasets.py --encoder nt_v2
uv run python scripts/train_logistic_probe.py --dataset tss_nt_v2_meanD --task family5

# 9. Cached paper analysis bundle: tables + figures + manifest
uv run python scripts/build_analysis_artifacts.py --overwrite
# Fast smoke version without UMAP:
uv run python scripts/build_analysis_artifacts.py --out /tmp/dna_analysis_smoke --skip-umap --overwrite
```

All caches (`data/sequences/`, `data/embeddings*/`, `data/chunk_reductions_*/`) are reused on rerun.

## Repository layout

```
src/                  Reusable Python packages (data_loader, splits, linear_trainer, kmer_baseline,
                      binary_tasks). One responsibility per package.
scripts/              CLI entrypoints — one script per experiment / run.
data/                 Artefacts. Small ones (metrics.json, splits.json, binary subset JSONs,
                      confusion matrices) are tracked; large parquets/embeddings are gitignored.
analysis/             Generated paper-ready analysis bundle from `scripts/build_analysis_artifacts.py`.
docs/superpowers/     Specs and implementation plans (`specs/`, `plans/`).
demo/                 Zero-shot demo: predicted family + neighbours for sample test genes.
viz/                  UMAP figures for the deck.
```

## Troubleshooting

- `pip: command not found`: use `uv run python ...` to run scripts and `uv pip ...` to manage packages inside the project environment.
- `uv pip install triton` fails on Apple Silicon + Python 3.12: this is expected here. Triton wheels are not available for this platform combination.
- DNABERT-2 on this repo does not actually require Triton for inference. The remote model code is supposed to fall back to plain PyTorch attention when Triton is missing.
- The runtime failure came from a `transformers` remote-code import check that treated DNABERT-2's optional `flash_attn_triton` module as mandatory. The loader now works around that and pins the model revision used by the repo.
- Do not try to speed this up by launching multiple concurrent encoder runs on the same `mps` or GPU device. Keep one encoder process per device; parallel fan-out is not the intended knob for this pipeline.
