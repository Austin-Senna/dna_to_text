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

# All four self-supervised encoders on TSS windows (RTX 5060: HyenaDNA
# ~80 min, DNABERT-2 ~110 min, GENA-LM ~70 min, NT-v2 already done).
for enc in nt_v2 dnabert2 gena_lm hyena_dna; do
  uv run python scripts/run_tss_multi_pool_extract.py --encoder "$enc" --device auto
  uv run python scripts/run_tss_probes_for_encoder.py --encoder "$enc" --skip-existing
done

# Enformer + TSS 4-mer baseline probes
uv run python scripts/train_logistic_probe.py --dataset enformer_tss_4mer --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_tss_4mer.parquet --probe-out data/probe_enformer_tss_4mer.npz
uv run python scripts/train_logistic_probe.py --dataset enformer_trunk_global --task family5
uv run python scripts/train_probe.py --dataset data/dataset_enformer_trunk_center.parquet --probe-out data/probe_enformer_trunk_center.npz

# Refresh bootstrap CIs over all 22 cells (12 CDS + 10 TSS)
uv run python scripts/bootstrap_test_uncertainty.py
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_enformer_features.py` | Runs Enformer on cached TSS windows and writes trunk/track feature datasets. |
| `scripts/run_tss_multi_pool_extract.py` | Runs DNA encoders over TSS windows and caches per-chunk reductions. |
| `scripts/build_tss_pooling_datasets.py` | Aggregates TSS per-chunk reductions into probe-ready datasets. |
| `scripts/train_logistic_probe.py` | Trains family5 probes for TSS-window feature sources. |
| `scripts/train_probe.py` | Trains Ridge-to-GenePT probes for TSS-window feature sources. |
| `src/data_loader/enformer_encoder.py` | Loads Enformer and extracts trunk/track summaries from TSS windows. |
| `src/data_loader/enformer_windows.py` | Supplies the cached 196,608 bp TSS windows. |
| `src/data_loader/multi_pool.py` | Shared chunked encoder extraction over long TSS windows. |
| `src/data_loader/pooling_aggregator.py` | Builds TSS pooling variants such as `meanmean`, `meanD`, and `meanG`. |
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

Every self-supervised encoder beats the TSS 4-mer baseline with
non-overlapping CIs (encoders recover non-trivial regulatory-context
signal), but no encoder is statistically separable from any other on
TSS. The substrate, not the encoder, is the dominant variable —
substrate dominance is encoder-general, not NT-v2-specific.
