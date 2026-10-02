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
data/                 Tracked split metadata (primary, seed and robustness splits),
                      the evaluation-purge pair table (data/leaks/), metrics,
                      confusion matrices, and selected feature/probe caches.
                      data/v2/ holds the camera-ready records of the canonical
                      run (committed); their stored predictions in
                      outputs/predictions/v2/ are not tracked.
analysis/             Generated diagnostics. analysis/figures/ and analysis/tables/
                      hold the retired Stage 6 outputs (May protocol, no
                      generator, removed at the camera-ready release), plus
                      per_dim_r2_distribution.png from scripts/per_dim_r2.py.
samples/              Small input/output examples for each pipeline stage.
dna_to_text_paper/    LaTeX manuscript source submodule for the report.
docs/                 Stage-level pipeline notes plus archived planning history.
tests/                Unit tests for artifact builders and encoder helpers.
```

Large intermediate caches such as fetched sequences, encoder chunk reductions, Enformer windows, and GenePT source artifacts are generated locally and ignored by git.

## Pipeline

The workflow has seven numbered stages. Stage 4 is the TSS branch: Stage 4.1 maps the CDS gene set to TSS-centered windows, and Stage 4.2 runs the TSS-window NT-v2 and Enformer comparisons. Stage 6 (the `analysis/` diagnostic artifacts) is retired; Stage 7 renders the manuscript figures and LaTeX table fragments under `dna_to_text_paper/paper/` (see `docs/stage7-paper-figures-tables.md`). Stage 7 reads the camera-ready records in `data/v2/`, which `scripts/recompute_all.sh` and `scripts/build_statistics.py` produce (the canonical records are committed; the stored predictions in `outputs/predictions/v2/` are not); full encoder extraction can take much longer and may require GPU or Apple Silicon MPS hardware.

Small sample inputs and outputs for each stage live in `samples/`. They are reviewer-readable examples of the data shape at each stage, not a separate lightweight execution path.

Report-supporting reproduction:

```bash
# Every probe cell and null band (Stages 3-5) from a committed, clean tree: records in
# data/v2/, stored test predictions in outputs/predictions/v2/. Resumable. To run in
# parallel, launch `scripts/recompute_all.sh all --shard I/N` for I = 0..N-1 first; the
# plain run below then only checks completeness, runs the G1 check, writes
# data/v2/run_complete.json, and runs the independent reimplementation of the headline
# cells (scripts/reproduce_headline.py -> data/v2/reproduction.json).
scripts/recompute_all.sh all

# Cluster-bootstrap intervals, paired tests, null bands and the sensitivity subsets ->
# data/v2/statistics.json, the rescored Ridge metrics (no refits), and the gene counts
# the paper states (single-chunk shares, noisy TF and kinase labels, templated summaries).
uv run python scripts/build_statistics.py
uv run python scripts/ridge_robust_metrics.py
uv run python scripts/build_counts.py     # also needs the CDS chunk caches from Stage 2 extraction
uv run python scripts/per_dim_r2.py

# Stage 7: regenerate the manuscript figures, LaTeX table fragments and prose numbers (see docs/stage7-paper-figures-tables.md).
uv run python scripts/build_result_figures.py
uv run python scripts/build_umap_compare.py
uv run python scripts/build_selection_sensitive.py
uv run python scripts/build_paper_tables.py
uv run python scripts/build_numbers.py --check
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

# Stage 3: make the splits. Every probe cell then runs through scripts/recompute_all.sh
# (above); probe fits refuse to run unless every BLAS/OpenMP pool is pinned to one
# thread (it changes lbfgs results). One cell by hand, for debugging:
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python scripts/make_splits.py
uv run python scripts/make_seed_splits.py   # seed splits 1, 7, 123 (CDS and TSS disjoint); make_splits.py writes seed 42 only
uv run python scripts/train_logistic_probe.py --dataset nt_v2_meanD --task family5

# Stage 4.1: canonical-TSS windows, the TSS split and matched TSS 4-mers
# (downloads the Ensembl 115 files first; see docs/stage4-1-tss-windows.md).
uv run python scripts/build_tss_windows.py
uv run python scripts/make_tss_disjoint_split.py
uv run python scripts/run_enformer_features.py --skip-model

# Stage 4.2: extract TSS-window features (each encoder, then Enformer); the TSS probe
# cells run in scripts/recompute_all.sh.
uv pip install ".[enformer]"
uv run python scripts/run_enformer_features.py --device auto
uv run python scripts/run_tss_multi_pool_extract.py --encoder nt_v2 --device auto
uv run python scripts/build_tss_pooling_datasets.py --encoder nt_v2

# Stages 3-5: every probe cell, then the cluster-bootstrap statistics (docs/stage5-bootstrap.md).
scripts/recompute_all.sh all
uv run python scripts/build_statistics.py
uv run python scripts/build_counts.py

# Stage 7: regenerate the manuscript figures, LaTeX table fragments and prose numbers.
uv run python scripts/build_result_figures.py
uv run python scripts/build_umap_compare.py
uv run python scripts/build_selection_sensitive.py
uv run python scripts/build_paper_tables.py
uv run python scripts/build_numbers.py --check
```

## Setup

Requires Python 3.11 or newer. The camera-ready records were fitted on Python 3.13.13, so reproduce them with `uv sync --python 3.13.13` (the repository pins no `.python-version`).

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
- Ensembl TSS windows: built by `scripts/build_tss_windows.py` from the Ensembl release 115 GTF, primary-assembly FASTA and cDNA FASTA (commands in `docs/stage4-1-tss-windows.md`); each window is 196,608 bp in gene orientation, centred on the 5' end of the gene's canonical transcript, and checked against the tracked manifest `data/tss_windows.tsv`.
- Encoder checkpoints: Hugging Face model IDs `zhihan1996/DNABERT-2-117M`, `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`, `AIRI-Institute/gena-lm-bert-base-t2t`, `LongSafari/hyenadna-large-1m-seqlen-hf` and `EleutherAI/enformer-official-rough`, each loaded at the commit pinned in `src/data_loader/model_registry.py`; every loader checks each weight against the checkpoint file. ESM-2 (fair-esm `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`) is pinned by checkpoint sha256 in the same file.

## Testing

pytest is a dev dependency (`uv sync` installs it); `tests/conftest.py` pins one thread and hides every GPU (`CUDA_VISIBLE_DEVICES=""`), so the suite runs on CPU and never borrows a shared card.

```bash
uv run pytest            # fast suite, a few seconds
uv run pytest -m slow    # real-data acceptance checks, a few minutes
```

## Troubleshooting

- `ThreadPinError` from a probe script: set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before Python starts. The thread count changes lbfgs results, so probe fits refuse to run unpinned. Even pinned, some family5 picks move with the OpenBLAS kernel; `data/v2/selection_sensitive.json` lists the cells that move under 6 threads or another kernel, and the paper marks each digit they move with a dagger.
- `pytest: No such file or directory`: run `uv sync` (pytest is a dev dependency), then `uv run pytest`.
- `pip: command not found`: use `uv run python ...` for scripts and `uv pip ...` to manage packages inside the project environment.
- `uv pip install triton` fails on Apple Silicon + Python 3.12: Triton wheels are not available for this platform combination, and DNABERT-2 inference in this repo does not require Triton.
- Encoder runs are expensive. Do not launch multiple concurrent encoder processes on the same MPS or GPU device; keep one encoder process per device.
