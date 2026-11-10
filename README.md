# MINA: a linear-probe benchmark for frozen DNA encoders

Code and records for *MINA* (MLCB 2026). MINA asks what the frozen embeddings of four DNA encoders (DNABERT-2, Nucleotide Transformer v2, GENA-LM, HyenaDNA) make linearly accessible about the protein a gene encodes, on 3,244 human genes from five HGNC families. Two linear probes read each embedding: 5-way family classification and ridge regression onto the gene's GenePT text-summary embedding. Every score is anchored by nucleotide and translated amino-acid composition, with ESM-2 as a protein-level upper bound and Enformer as a supervised comparator on TSS windows, and the evaluation varies three choices the paper shows matter as much as the encoder: the pooling rule, the substrate (CDS or a 196,608 bp TSS window) and the split (homology-aware, interval-disjoint or random).

Paper: `dna_to_text.pdf`. The tag `mlcb2026-camera-ready` marks the code behind every number in it.

## Repository layout

```
src/                Python packages: data loading, splits, encoder wrappers, pooling,
                    composition baselines, the single probe fit path.
scripts/            CLI entrypoints for data prep, encoder extraction, probes, bootstrap
                    statistics and the paper's figures, tables and prose numbers.
data/               Splits (primary, seed and robustness), cluster assignments, the
                    evaluation-purge pair table (data/leaks/), window and CDS manifests,
                    the pooled feature parquets, and data/v2/: the camera-ready records
                    of every probe cell and the bootstrap statistics the paper reports.
analysis/           Generated diagnostics: the TSS-window audits (analysis/tss_overlap/).
docs/examples/      Small input/output examples for each pipeline stage.
docs/               Stage-level pipeline notes and the Zenodo deposit description.
tests/              Unit and acceptance tests.
```

The input sequences, the gene table, per-chunk feature caches, Enformer and ESM-2 features and the stored test predictions are not in git. They are deposited on Zenodo (DOI `10.5281/zenodo.23031890`); every tarball unpacks at the repository root, and `docs/zenodo_deposit.md` lists them. With the deposit unpacked, every step after embedding extraction runs on CPU.

## Setup

Requires Python 3.11 or newer. The camera-ready records were fitted on Python 3.13.13, so reproduce them with `uv sync --frozen --python 3.13.13`.

```bash
git clone https://github.com/Austin-Senna/dna_to_text
uv sync --frozen
uv pip install ".[enformer]"   # optional: the Enformer comparator
```

Use `uv run ...` for commands unless `.venv/` is activated.

## Reproducing the paper's numbers

From the committed records in `data/v2/`:

```bash
uv run python scripts/build_statistics.py        # cluster-bootstrap intervals, paired tests, null bands
uv run python scripts/build_selection_sensitive.py
uv run python scripts/build_result_figures.py    # figures
uv run python scripts/build_paper_tables.py      # LaTeX table fragments
uv run python scripts/build_numbers.py --check   # every number the prose prints
uv run python scripts/build_fig1.py --check      # Figure 1's numbers; without --check it re-renders (Docker)
```

The figure, table and number builders write into the paper's LaTeX source tree at `dna_to_text_paper/paper/`, which is not part of this repository.

Refitting every probe cell from the deposit (CPU, resumable; records land in `data/v2/`, stored predictions in `outputs/predictions/v2/`):

```bash
scripts/recompute_all.sh all        # or shard: scripts/recompute_all.sh all --shard I/N
uv run python scripts/build_statistics.py
uv run python scripts/ridge_robust_metrics.py
uv run python scripts/build_counts.py
```

`scripts/reproduce_headline.py` is an independent minimal reimplementation (NumPy, pandas, scikit-learn) of the headline cells; `recompute_all.sh` runs it last and writes `data/v2/reproduction.json`.

Probe fits refuse to run unless every BLAS/OpenMP pool is pinned to one thread, because the thread count changes lbfgs results:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Even pinned, a few family-classification picks move with the OpenBLAS kernel; `data/v2/selection_sensitive.json` lists them and the paper marks their digits with a dagger.

## Rebuilding from public sources

Embedding extraction needs a GPU (or Apple Silicon MPS). The stages, each described in `docs/`:

```bash
# Stage 1: gene table and canonical Ensembl CDS
uv run python scripts/prepare_data.py

# Stage 2: CDS embeddings and pooled feature datasets
uv run python scripts/run_encoder.py --device auto
uv run python scripts/run_nt_v2_encoder.py --device auto
for enc in dnabert2 nt_v2 gena_lm hyena_dna; do
  uv run python scripts/run_multi_pool_extract.py --encoder $enc
  uv run python scripts/build_pooling_datasets.py --encoder $enc
done
uv run python scripts/run_esm2.py

# Stage 3: splits (seed 42, then seeds 1, 7, 123)
uv run python scripts/make_splits.py
uv run python scripts/make_seed_splits.py

# Stage 4.1: canonical-TSS windows, the interval-disjoint split, matched TSS 4-mers
uv run python scripts/build_tss_windows.py
uv run python scripts/make_tss_disjoint_split.py
uv run python scripts/run_enformer_features.py --skip-model
uv run python scripts/tss_overlap.py

# Stage 4.2: TSS-window features, each encoder and Enformer
uv run python scripts/run_enformer_features.py --device auto
uv run python scripts/run_tss_multi_pool_extract.py --encoder nt_v2 --device auto
uv run python scripts/build_tss_pooling_datasets.py --encoder nt_v2

# Stages 3-5: every probe cell, then statistics (see above)
scripts/recompute_all.sh all
```

External inputs:

- GenePT v2 embeddings: Zenodo `10.5281/zenodo.10833191`; unzip `GenePT_emebdding_v2.zip` into `GenePT_emebdding_v2/`.
- HGNC complete gene set: fetched by `src/data_loader/dataset_loader.py`.
- Ensembl canonical CDS: fetched by `src/data_loader/sequence_fetcher.py` from the Ensembl REST API.
- Ensembl release 115 GTF, primary-assembly FASTA and cDNA FASTA for the TSS windows (`docs/stage4-1-tss-windows.md`); the tracked manifest `data/tss_windows.tsv` pins every window's sha256.
- Encoder checkpoints: `zhihan1996/DNABERT-2-117M`, `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`, `AIRI-Institute/gena-lm-bert-base-t2t`, `LongSafari/hyenadna-large-1m-seqlen-hf`, `EleutherAI/enformer-official-rough`, each at the commit pinned in `src/data_loader/model_registry.py`; ESM-2 (`esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`) pinned by checkpoint sha256 in the same file.

## Tests

```bash
uv run pytest            # fast suite
uv run pytest -m slow    # real-data acceptance checks
```

`tests/conftest.py` pins one thread and hides every GPU, so the suite runs on CPU. Tests that read the gene table or the cached sequences need the Zenodo deposit unpacked at the repository root; tests of the paper builders (`test_build_numbers.py`, `test_fig1.py`) need the paper's LaTeX source at `dna_to_text_paper/`.

## Licences

By path (`LICENSES.md`): code MIT (`LICENSE`); data and results CC BY 4.0; files derived from NT-v2 outputs CC BY-NC-SA 4.0, NT-v2's own licence.
