# dna_to_text

Cross-modal probing of frozen DNA sequence representations against gene-family labels and GenePT text embeddings. The paper compares DNABERT-2, Nucleotide Transformer v2, GENA-LM, and HyenaDNA on a 3,244-gene 5-family classification task, with Enformer reported separately as a supervised sequence-to-function comparator.

Project repository: https://github.com/Austin-Senna/dna_to_text

Final report PDF: `dna_to_text.pdf`

Submission-facing notes are in `submission.md`.
Additional documentation is indexed in `docs/README.md`.

## Repository Layout

```
src/                  Reusable Python packages for data loading, split handling,
                      encoder wrappers, pooling, baselines, and linear probes.
scripts/              CLI entrypoints for data prep, encoder extraction, probes,
                      baselines, bootstrap uncertainty, and artifact generation.
data/                 Tracked split metadata, metrics, confusion matrices,
                      selected feature/probe caches, and bootstrap outputs.
analysis/             Report-ready generated figures, tables, and visualizations.
  figures/            Main and supplementary report figures.
  tables/             CSV and Markdown tables used by the report.
samples/              Small input/output examples for each pipeline stage.
dna_to_text_paper/    LaTeX manuscript source submodule for the report.
docs/                 Stage-level pipeline notes plus archived planning history.
tests/                Unit tests for artifact builders and encoder helpers.
```

Large intermediate caches such as fetched sequences, encoder chunk reductions, Enformer windows, and GenePT source artifacts are generated locally and ignored by git.

## Pipeline

The workflow has six numbered stages. Stage 4 is the TSS branch: Stage 4.1 maps the CDS gene set to TSS-centered windows, and Stage 4.2 runs the TSS-window NT-v2 and Enformer comparisons. Most report-facing commands operate on tracked caches; full encoder extraction can take much longer and may require GPU or Apple Silicon MPS hardware.

Small sample inputs and outputs for each stage live in `samples/`. They are reviewer-readable examples of the data shape at each stage, not a separate lightweight execution path.

Report-supporting cached reproduction:

```bash
# Stage 5: regenerate 1000-run bootstrap confidence intervals (see docs/stage5-bootstrap.md).
uv run python scripts/bootstrap_test_uncertainty.py

# Stage 6: regenerate report tables and figures from tracked metrics/caches.
uv run python scripts/build_analysis_artifacts.py --overwrite
```

Fast Stage 6 smoke version without UMAP:

```bash
uv run python scripts/build_analysis_artifacts.py --out /tmp/dna_analysis_smoke --skip-umap --overwrite
```

Full data/encoder pipeline, when rebuilding from public sources:

```bash
# Stage 1: build the gene table and fetch canonical Ensembl CDS.
uv run python scripts/prepare_data.py

# Stage 2: extract CDS embeddings and materialize pooled feature datasets.
uv run python scripts/run_encoder.py --device auto
uv run python scripts/run_nt_v2_encoder.py --device auto
uv run python scripts/run_multi_pool_extract.py --encoder dnabert2
uv run python scripts/run_multi_pool_extract.py --encoder nt_v2
uv run python scripts/run_multi_pool_extract.py --encoder gena_lm
uv run python scripts/run_multi_pool_extract.py --encoder hyena_dna
uv run python scripts/build_pooling_datasets.py --encoder dnabert2
uv run python scripts/build_pooling_datasets.py --encoder nt_v2
uv run python scripts/build_pooling_datasets.py --encoder gena_lm
uv run python scripts/build_pooling_datasets.py --encoder hyena_dna

# Stage 3: make frozen splits and train headline probes/baselines.
uv run python scripts/make_splits.py
uv run python scripts/train_logistic_probe.py --dataset nt_v2_meanD --task family5
uv run python scripts/train_probe.py --dataset data/dataset_dnabert2_meanG.parquet
uv run python scripts/build_family5_table.py
uv run python scripts/build_regression_table.py

# Stage 4.1: map the CDS gene set to TSS-centered windows and matched TSS 4-mers.
uv run python scripts/run_enformer_features.py --skip-model

# Stage 4.2: run TSS-window NT-v2 and Enformer, then train context-ablation probes.
uv pip install ".[enformer]"
uv run python scripts/run_enformer_features.py --device auto
uv run python scripts/run_tss_multi_pool_extract.py --encoder nt_v2 --device auto
uv run python scripts/build_tss_pooling_datasets.py --encoder nt_v2
uv run python scripts/train_logistic_probe.py --dataset enformer_tss_4mer --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_tss_4mer.parquet --probe-out data/probe_enformer_tss_4mer.npz
uv run python scripts/train_logistic_probe.py --dataset tss_nt_v2_meanmean --task family5
uv run python scripts/train_probe.py --dataset data/dataset_tss_nt_v2_meanmean.parquet --probe-out data/probe_tss_nt_v2_meanmean.npz
uv run python scripts/train_logistic_probe.py --dataset enformer_trunk_global --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_trunk_center.parquet --probe-out data/probe_enformer_trunk_center.npz

# Stage 5: regenerate 1000-run bootstrap confidence intervals (see docs/stage5-bootstrap.md).
uv run python scripts/bootstrap_test_uncertainty.py

# Stage 6: regenerate final report analysis tables and figures.
uv run python scripts/build_analysis_artifacts.py --overwrite
```

## Setup

Requires Python 3.11 or newer.

```bash
uv sync
```

Use `uv run ...` for commands unless you have already activated `.venv/`.

Optional Enformer comparator dependency:

```bash
uv pip install ".[enformer]"
```

External large inputs:

- GenePT v2 artifacts: Zenodo DOI `10.5281/zenodo.10833191`; unzip `GenePT_emebdding_v2.zip` into `GenePT_emebdding_v2/`.
- HGNC complete gene set: downloaded by `src/data_loader/dataset_loader.py` from `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt`.
- Ensembl canonical CDS: fetched by `src/data_loader/sequence_fetcher.py` from Ensembl REST `/lookup/id/{gene_id}` and `/sequence/id/{transcript_id}?type=cds`.
- Ensembl TSS windows: derived by `src/data_loader/enformer_windows.py` from Ensembl REST gene coordinates (`/lookup/id/{gene_id}`); each window is 196,608 bp centered on the strand-aware gene TSS and fetched from `/sequence/region/human/{region}`.
- Encoder checkpoints: Hugging Face model IDs `zhihan1996/DNABERT-2-117M`, `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`, `AIRI-Institute/gena-lm-bert-base-t2t`, and `LongSafari/hyenadna-large-1m-seqlen-hf`.

## Testing

```bash
uv run python -m unittest
```

If `pytest` is available in your environment, the same tests can also be run with:

```bash
uv run pytest
```

## Troubleshooting

- `pytest: No such file or directory`: this project uses `unittest` tests and does not require pytest by default. Use `uv run python -m unittest`, or install pytest in your environment if you prefer that runner.
- `pip: command not found`: use `uv run python ...` for scripts and `uv pip ...` to manage packages inside the project environment.
- `uv pip install triton` fails on Apple Silicon + Python 3.12: Triton wheels are not available for this platform combination, and DNABERT-2 inference in this repo does not require Triton.
- Encoder runs are expensive. Do not launch multiple concurrent encoder processes on the same MPS or GPU device; keep one encoder process per device.
