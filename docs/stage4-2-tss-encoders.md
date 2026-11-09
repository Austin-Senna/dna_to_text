# Stage 4.2: Run TSS Encoders and Context Ablation

Stage 4.2 runs the TSS-window context comparison. It evaluates matched TSS
4-mer features, TSS-window features from all four self-supervised encoders
(NT-v2, DNABERT-2, GENA-LM, HyenaDNA), and Enformer trunk features against
the same family-classification and GenePT-regression targets.

## Sample Files

- Input: `samples/stage4_2_tss_encoder_input.json`
- Output: `samples/stage4_2_tss_encoder_output.json`

## Full Commands

```bash
uv pip install ".[enformer]"
uv run python scripts/run_enformer_features.py --device auto
# Probe fits refuse to run unless every BLAS/OpenMP pool is pinned to one thread.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# All four self-supervised encoders on the canonical-TSS windows (Stage 4.1).
# May timings on an RTX 5060: HyenaDNA ~80 min, DNABERT-2 ~110 min, GENA-LM ~70 min.
# On a card with little VRAM, run_tss_extract_capped.py takes the same arguments.
for enc in nt_v2 dnabert2 gena_lm hyena_dna; do
  uv run python scripts/run_tss_multi_pool_extract.py --encoder "$enc" --device auto
  uv run python scripts/build_tss_pooling_datasets.py --encoder "$enc"
done
# E5 features from the cached chunks (no GPU): TSS-anchored pools and their
# chunk-matched composition baselines
uv run python scripts/build_tss_anchored_datasets.py
uv run python scripts/build_tss_composition_baseline.py

# Per-cell probes default to data/splits.json; the TSS primary must be named.
# Probe each pool in the encoder's TSS grid (model_registry.encoder_pools(enc, "TSS")):
S=data/splits_tss_disjoint.json
uv run python scripts/train_logistic_probe.py --dataset tss_nt_v2_meanmean --task family5 --splits "$S"

# Enformer + TSS 4-mer baseline probes
uv run python scripts/train_logistic_probe.py --dataset enformer_tss_4mer --task family5 --splits "$S"
uv run python scripts/train_probe.py --dataset data/dataset_enformer_tss_4mer.parquet --splits "$S" --probe-out data/probe_enformer_tss_4mer.npz
uv run python scripts/train_logistic_probe.py --dataset enformer_trunk_global --task family5 --splits "$S"
uv run python scripts/train_probe.py --dataset data/dataset_enformer_trunk_global.parquet --splits "$S" --probe-out data/probe_enformer_trunk_global.npz

# Camera-ready: every CDS and TSS cell, then the intervals (rescores stored predictions)
scripts/recompute_all.sh all
uv run python scripts/build_statistics.py
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_enformer_features.py` | Runs Enformer on cached TSS windows and writes trunk/track feature datasets; `--no-datasets` only fills the feature cache, `--from-cache` builds the datasets from a finished cache (e.g. one extracted on another machine) without the model. |
| `scripts/run_tss_multi_pool_extract.py` | Runs DNA encoders over TSS windows and caches per-chunk reductions; `--gene-table` restricts the run to a parquet's genes. It calls `embed_all_multi_pool` with `collect=False`, so the reductions go to the per-gene `.npz` cache and are not held in RAM. |
| `scripts/run_tss_extract_capped.py` | VRAM-capped launcher for the TSS extractor: caps the process at `MEM_FRACTION` of GPU memory (default 0.55) and sets `PYTORCH_CUDA_ALLOC_CONF` before the encoder loads, so a memory spike raises a Python OOM and a rerun resumes from the cache instead of crashing the driver. Forwards every argument to `run_tss_multi_pool_extract.py`. |
| `scripts/check_tss_center_chunk.py` | Tokenizer-only check of where each encoder's middle chunk falls relative to the TSS; writes `analysis/tss_overlap/center_chunk_offsets_<encoder>.csv`, which the two E5 builders below read. |
| `scripts/build_tss_anchored_datasets.py` | E5 `tssanchored` pool: per gene, the cached chunk whose content is most centred on the TSS. No GPU. |
| `scripts/build_tss_composition_baseline.py` | E5 chunk-matched composition baselines (`chunk4mergc`, `chunk6mer`) over the same anchor chunk each encoder pooled. No GPU. |
| `scripts/aws/extract_box.sh` | Runs every GPU extraction (CDS, TSS, Enformer, ESM-2) on one CUDA box, in order, and records the GPU, driver and torch build; `pilot` first replays the pre-fix code on a few genes whose inputs did not change and checks that the new code reproduces it bit for bit (HyenaDNA excepted: it now runs without CLS/SEP, so it is checked for constant features instead; GENA-LM excepted: the pre-fix code ran it at its random initialisation, so only the run-to-run repeat gates it). |
| `scripts/compare_extraction_caches.py` | `pick` chooses the pilot genes; `compare` checks two caches gene by gene (bit-exact or against error thresholds); `census` checks that a finished cache holds exactly the manifest's genes, all stamped, finite and from a single run. |
| `scripts/build_tss_pooling_datasets.py` | Aggregates TSS per-chunk reductions into probe-ready datasets. |
| `scripts/train_logistic_probe.py` | Trains family5 probes for TSS-window feature sources. |
| `scripts/train_probe.py` | Trains Ridge-to-GenePT probes for TSS-window feature sources. |
| `src/data_loader/enformer_encoder.py` | Loads Enformer at its pinned revision and extracts trunk/track summaries. `trunk_global` averages all 896 output bins, the central 114,688 bp of the window; `trunk_center` the central 16 bins (2,048 bp). |
| `src/data_loader/enformer_windows.py` | Supplies the canonical-TSS windows through `read_window`, which checks each against the manifest (Stage 4.1). |
| `src/data_loader/multi_pool.py` | Shared chunked encoder extraction over long TSS windows; caches in `data/tss_chunk_reductions_v2_<encoder>/` carry a meta record and are refused if built from other windows or another revision, or if they are unstamped or mix GPUs or torch builds (a cache made on another machine is accepted). |
| `src/data_loader/pooling_aggregator.py` | Builds TSS pooling variants: `meanmean`, `maxmean`, `clsmean`, `meanD`, `meanG` (HyenaDNA has no `clsmean`), plus the E5 `centermean` template. |
| `samples/stage4_2_tss_encoder_input.json` | Tiny example of context-ablation feature sources and commands. |
| `samples/stage4_2_tss_encoder_output.json` | Tiny excerpt of the CDS-vs-TSS result table. |

## Outputs

- `data/dataset_tss_<encoder>_<pool>.parquet` - TSS-window pooling datasets for all four encoders.
- `data/dataset_tss_<encoder>_{tssanchored,chunk4mergc,chunk6mer}.parquet` - the E5 anchored pools and their chunk-matched composition baselines.
- `data/dataset_enformer_trunk_global.parquet`, `data/dataset_enformer_trunk_center.parquet` - Enformer trunk features: the whole-window readout, and the central 16 bins (the E5 counterpart). Both run on both tasks.
- The CDS vs TSS comparison is built in Stage 7 (`cds_tss.tex`, `tss_context.png`) from the `data/v2/` records. `analysis/tables/context_ablation.md` and `analysis/figures/context_ablation_cds_tss_enformer.png` are retired Stage 6 outputs (May protocol, no generator).

## Splits

Every TSS result is reported on `data/splits_tss_disjoint.json` (Stage 4.1).
`scripts/recompute_all.py` runs the full TSS grid, E5 cells included, on that
split and on `data/splits.json` (the window-overlap sensitivity analysis), and
the whole-window grid on the random split and the three disjoint seed splits.

## Results

This page states no results. The current numbers live in the records
`data/v2/metrics_splits_tss_disjoint.json` (and the other `data/v2/metrics_*.json`)
and in `data/v2/statistics.json`. The paper's tables come from
`scripts/build_paper_tables.py` (the substrate ablation is `cds_tss.tex`), and its
prose numbers (`numbers.tex`) from `scripts/build_numbers.py`. To regenerate them,
run `scripts/recompute_all.sh all` (see its header), then the builders in
`docs/stage7-paper-figures-tables.md`.
