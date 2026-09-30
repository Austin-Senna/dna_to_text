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
for enc in nt_v2 dnabert2 gena_lm hyena_dna; do
  uv run python scripts/run_tss_multi_pool_extract.py --encoder "$enc" --device auto
  uv run python scripts/build_tss_pooling_datasets.py --encoder "$enc"
done
# Probe each pool in the encoder's TSS grid (model_registry.encoder_pools(enc, "TSS")):
uv run python scripts/train_logistic_probe.py --dataset tss_nt_v2_meanmean --task family5 \
  --splits data/splits_tss_disjoint.json

# Enformer + TSS 4-mer baseline probes
uv run python scripts/train_logistic_probe.py --dataset enformer_tss_4mer --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_tss_4mer.parquet --probe-out data/probe_enformer_tss_4mer.npz
uv run python scripts/train_logistic_probe.py --dataset enformer_trunk_global --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_trunk_center.parquet --probe-out data/probe_enformer_trunk_center.npz

# Refresh bootstrap CIs over the CDS and TSS headline cells (rescores stored predictions)
uv run python scripts/bootstrap_test_uncertainty.py
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_enformer_features.py` | Runs Enformer on cached TSS windows and writes trunk/track feature datasets; `--no-datasets` only fills the feature cache, `--from-cache` builds the datasets from a finished cache (e.g. one extracted on another machine) without the model. |
| `scripts/run_tss_multi_pool_extract.py` | Runs DNA encoders over TSS windows and caches per-chunk reductions; `--gene-table` restricts the run to a parquet's genes. |
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

- `data/dataset_tss_nt_v2_*.parquet` - TSS-window NT-v2 pooling datasets.
- `data/dataset_enformer_trunk_global.parquet` - Enformer trunk family5 feature table.
- `data/dataset_enformer_trunk_center.parquet` - Enformer trunk regression feature table.
- `analysis/tables/context_ablation.md` - CDS vs TSS context comparison.
- `analysis/figures/context_ablation_cds_tss_enformer.png` - report figure.

## Result Summary

All four self-supervised encoders collapse from CDS to TSS into a tight
macro-F1 band of 0.39--0.46, with mutually-overlapping 95% bootstrap CIs:

| Encoder | TSS best pool | macro-F1 [95% CI] | TSS R² [95% CI] |
| --- | --- | --- | --- |
| 4-mer (TSS) | — | 0.247 [0.225, 0.268] | 0.041 [0.028, 0.050] |
| GENA-LM | clsmean | 0.389 [0.331, 0.442] | 0.059 [0.042, 0.071] |
| HyenaDNA | meanmean | 0.419 [0.356, 0.476] | 0.085 [0.065, 0.101] |
| NT-v2 | meanmean | 0.447 [0.384, 0.507] | 0.117 [0.094, 0.137] |
| DNABERT-2 | maxmean | 0.455 [0.394, 0.517] | 0.122 [0.100, 0.140] |
| Enformer trunk | center | 0.545 | 0.142 |

Note (Sept 29, 2026): the GENA-LM row came from a randomly initialised network (a loader bug, fixed in
`7a0c6e1`), not the pretrained encoder. This table predates the camera-ready re-extraction, which replaces it.

Every self-supervised encoder beats the TSS 4-mer baseline with
non-overlapping CIs (encoders recover non-trivial regulatory-context
signal), but no encoder is statistically separable from any other on
TSS. The substrate, not the encoder, is the dominant variable —
substrate dominance is encoder-general, not NT-v2-specific.
