# Stage 2: Encode CDS and Pool Features

Stage 2 converts canonical CDS sequences into frozen DNA-encoder feature
tables. It runs the pretrained encoders, caches per-gene or per-chunk
reductions, and materializes pooling variants used by the probes.

## Sample Files

- Input: `samples/stage2_cds_input.fasta`
- Output: `samples/stage2_encoder_output.json`

## Full Commands

```bash
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
```

## Encoder Checkpoints

- DNABERT-2: `zhihan1996/DNABERT-2-117M`
- Nucleotide Transformer v2: `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`
- GENA-LM: `AIRI-Institute/gena-lm-bert-base-t2t`
- HyenaDNA: `LongSafari/hyenadna-large-1m-seqlen-hf`

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_encoder.py` | Legacy DNABERT-2 single-vector extraction entrypoint. |
| `scripts/run_nt_v2_encoder.py` | Legacy NT-v2 single-vector extraction entrypoint. |
| `scripts/run_multi_pool_extract.py` | Extracts per-chunk reductions used to build pooling variants. |
| `scripts/build_pooling_datasets.py` | Aggregates cached reductions into probe-ready parquet datasets. |
| `src/data_loader/model_registry.py` | Central registry of encoder names, cache names, dimensions, and loader modules. |
| `src/data_loader/encoder_runner.py` | DNABERT-2 model loading and CDS embedding helpers. |
| `src/data_loader/nt_v2_encoder.py` | NT-v2 model loading and embedding helpers. |
| `src/data_loader/gena_lm_encoder.py` | GENA-LM model loading and embedding helpers. |
| `src/data_loader/hyena_dna_encoder.py` | HyenaDNA model loading and embedding helpers. |
| `src/data_loader/multi_pool.py` | Shared per-chunk embedding loop for pooling variants. |
| `src/data_loader/pooling_aggregator.py` | Converts per-chunk reductions into fixed-length pooling variants. |
| `samples/stage2_cds_input.fasta` | Tiny CDS FASTA example entering encoder extraction. |
| `samples/stage2_encoder_output.json` | Tiny example of pooled feature metadata and vector shape. |

## Outputs

- `data/dataset_<encoder>_<pooling>.parquet` - probe-ready feature tables.
- `data/chunk_reductions_<encoder>/` - ignored local per-gene reduction caches.

Encoder extraction is the expensive stage. Use one GPU/MPS encoder process per
device and rely on caches for interrupted reruns.
