# Stage 4.2: Run TSS Encoders and Context Ablation

Stage 4.2 runs the TSS-window context comparison. It evaluates matched TSS
4-mer features, TSS-window NT-v2 features, and Enformer trunk features against
the same family-classification and GenePT-regression targets.

## Sample Files

- Input: `samples/stage4_2_tss_encoder_input.json`
- Output: `samples/stage4_2_tss_encoder_output.json`

## Full Commands

```bash
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

CDS NT-v2 remains strongest for this protein-family task. TSS-window NT-v2 and
Enformer are informative but lower than the CDS signal, supporting the
interpretation that the target labels are mostly coding-sequence and
protein-domain signals rather than promoter-context signals.
