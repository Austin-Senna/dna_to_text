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

Every loader compares each loaded weight with the tensor in the checkpoint file. GENA-LM needs it: under
transformers 5.5.4 its `from_pretrained` reported every weight loaded but left all of them at their random
initialisation, so GENA-LM features extracted before commit `7a0c6e1` came from an untrained network.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_encoder.py` | Legacy DNABERT-2 single-vector extraction entrypoint. |
| `scripts/run_nt_v2_encoder.py` | Legacy NT-v2 single-vector extraction entrypoint. |
| `scripts/run_multi_pool_extract.py` | Extracts per-chunk reductions used to build pooling variants; `--cache-dir` writes elsewhere than the encoder's default (the AWS pilot). |
| `scripts/build_pooling_datasets.py` | Aggregates cached reductions into probe-ready parquet datasets. |
| `scripts/run_esm2.py` | Embeds full-length proteins with ESM-2 (`--size 150m` or `650m`), fp32; `--fp16` exists only to replay the May fp16 embeddings. The checkpoint must match its pinned sha256. |
| `scripts/build_esm2_datasets.py` | Builds `data/dataset_esm2_<size>.parquet` from `data/esm2_<size>_embeddings_v2/`; refuses fp16, unpinned, unstamped or mixed-run embeddings. |
| `scripts/aws/extract_box.sh`, `scripts/compare_extraction_caches.py` | GPU-box runner and cache checks for the CDS and TSS extractions (see Stage 4.2). |
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
- `data/chunk_reductions_v2_<encoder>/` - ignored local per-gene reduction caches, each with a meta record (encoder revision, chunking, boundary tokens, device, GPU and torch build, input sha256); a cache built differently is refused, and the dataset builders refuse a cache that mixes GPUs or torch builds, or lacks the runtime stamp.

HyenaDNA is run on DNA tokens only (no CLS/SEP): it was never trained with a
CLS token, and as a causal model a CLS at position 0 reaches every position.
It therefore has no `clsmean` or `specialmean` pools, and the CDS grid has 22
encoder x pooling configs.

Encoder extraction is the expensive stage. Use one GPU/MPS encoder process per
device and rely on caches for interrupted reruns: `embed_all_multi_pool` in
`src/data_loader/multi_pool.py` writes one `.npz` per gene and skips any gene whose
cached meta matches. For the long TSS windows (Stage 4.2) it runs with
`collect=False`, which keeps reductions out of RAM, and
`scripts/run_tss_extract_capped.py` caps GPU memory so an out-of-memory error stops
the run cleanly instead of crashing the driver.
