# Run NT-v2 On TSS Windows In Google Colab A100

This is the Colab path for the CDS-vs-TSS ablation:

- input: 196,608 bp TSS-centered genomic windows
- encoder: NT-v2 100M multi-species
- outputs: `dataset_tss_nt_v2_*.parquet`, family5 metrics, Ridge probes, confusion JSONs

The scripts cache one `.npz` per gene, so the extraction is resumable. If Colab
disconnects, rerun the same extraction command and it will skip cached genes.

Prefer an importable notebook? Use `notebooks/nt_v2_tss_a100_colab.ipynb`.

## 0. Start The Runtime

In Colab:

1. `Runtime -> Change runtime type`
2. Hardware accelerator: `A100 GPU`
3. Runtime shape: high-RAM if available

Then run:

```python
!nvidia-smi

import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
```

## 1. Clone The Branch

Use the active branch until this work is merged:

```python
!git clone -b codex/dna-encoder-expansion https://github.com/Austin-Senna/dna_to_text.git
%cd dna_to_text
```

If you already cloned it:

```python
%cd /content/dna_to_text
!git pull
```

## 2. Install Dependencies

Colab already ships with a CUDA-enabled PyTorch. This install keeps that and
adds the repo dependencies.

```python
!pip install -q -U pip
!pip install -q -e .
```

## 3. Restore TSS Windows Cache From Drive

Upload `tss_windows_cache.tar.gz` to Google Drive first. The cache bundle should
contain `enformer_windows/` and ideally `dataset_enformer_tss_4mer.parquet` at
the archive root; these extract under `data/` in Colab. If this restores `3244`
FASTA files, skip the Ensembl fetch step entirely.

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
from pathlib import Path

%cd /content/dna_to_text
shutil.rmtree("data/enformer_windows", ignore_errors=True)
Path("data/dataset_enformer_tss_4mer.parquet").unlink(missing_ok=True)
!tar -xzf /content/drive/MyDrive/tss_windows_cache.tar.gz -C data
!find data/enformer_windows -maxdepth 1 -name '*.fa' | wc -l
!ls -lh data/dataset_enformer_tss_4mer.parquet
```

Optional but useful for Hugging Face cache persistence:

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/hf_cache"
```

## 4. Optional Fallback: Fetch/Cache TSS Windows

Only use this if you do not have `tss_windows_cache.tar.gz`. This builds
`data/enformer_windows/` and rewrites the matched TSS 4-mer parquet. It does
not load Enformer when `--skip-model` is set.

```python
!python scripts/run_enformer_features.py \
  --template-dataset data/dataset_nt_v2_meanD.parquet \
  --skip-model
```

Quick sanity check:

```python
!find data/enformer_windows -maxdepth 1 -name '*.fa' | wc -l
!ls -lh data/dataset_enformer_tss_4mer.parquet
```

Expected window count: `3244`.

## 5. Run NT-v2 TSS Extraction

Full run:

```python
!python scripts/run_tss_multi_pool_extract.py \
  --encoder nt_v2 \
  --template-dataset data/dataset_nt_v2_meanD.parquet \
  --device cuda
```

Small pilot first:

```python
!python scripts/run_tss_multi_pool_extract.py \
  --encoder nt_v2 \
  --template-dataset data/dataset_nt_v2_meanD.parquet \
  --device cuda \
  --max-genes 25
```

If the pilot looks good, rerun the full command without `--max-genes`. Cached
pilot genes will be skipped.

Sanity check:

```python
!find data/tss_chunk_reductions_nt_v2 -maxdepth 1 -name '*.npz' | wc -l
!du -sh data/tss_chunk_reductions_nt_v2
```

Expected final count: `3244`.

## 6. Build TSS Pooling Datasets

```python
!python scripts/build_tss_pooling_datasets.py \
  --encoder nt_v2 \
  --template-dataset data/dataset_nt_v2_meanD.parquet \
  --variants meanmean meanD meanG maxmean clsmean
```

This writes:

```text
data/dataset_tss_nt_v2.parquet
data/dataset_tss_nt_v2_meanmean.parquet
data/dataset_tss_nt_v2_meanD.parquet
data/dataset_tss_nt_v2_meanG.parquet
data/dataset_tss_nt_v2_maxmean.parquet
data/dataset_tss_nt_v2_clsmean.parquet
```

## 7. Run Family5 And Ridge Probes

Run the main pooling variants first:

```bash
%%bash
set -euo pipefail

for v in meanmean meanD meanG; do
  python scripts/train_logistic_probe.py --dataset "tss_nt_v2_${v}" --task family5
  python scripts/train_probe.py \
    --dataset "data/dataset_tss_nt_v2_${v}.parquet" \
    --probe-out "data/probe_tss_nt_v2_${v}.npz"
done
```

Optional full pooling sweep:

```bash
%%bash
set -euo pipefail

for v in maxmean clsmean; do
  python scripts/train_logistic_probe.py --dataset "tss_nt_v2_${v}" --task family5
  python scripts/train_probe.py \
    --dataset "data/dataset_tss_nt_v2_${v}.parquet" \
    --probe-out "data/probe_tss_nt_v2_${v}.npz"
done
```

## 8. Rebuild Tables

```python
!python scripts/build_family5_table.py
!python scripts/build_regression_table.py
```

Inspect the TSS sections:

```python
!sed -n '/TSS Self-Supervised Encoder Ablation/,$p' data/family5_table.md | head -n 20
!sed -n '/TSS Self-Supervised Encoder Ablation/,$p' data/regression_table.md | head -n 24
```

## 9. Package Outputs

Minimal bundle to bring back to the repo:

```python
!tar -czf ntv2_tss_minimal_outputs.tar.gz \
  data/dataset_tss_nt_v2*.parquet \
  data/probe_tss_nt_v2*.npz \
  data/confusion_5way_tss_nt_v2*.json \
  data/metrics.json \
  data/family5_table.md \
  data/regression_table.md
```

Full resumable cache bundle:

```python
!tar -czf ntv2_tss_full_cache_outputs.tar.gz \
  data/tss_chunk_reductions_nt_v2 \
  data/dataset_tss_nt_v2*.parquet \
  data/probe_tss_nt_v2*.npz \
  data/confusion_5way_tss_nt_v2*.json \
  data/metrics.json \
  data/family5_table.md \
  data/regression_table.md
```

Copy to Google Drive:

```python
!cp ntv2_tss_minimal_outputs.tar.gz /content/drive/MyDrive/
!cp ntv2_tss_full_cache_outputs.tar.gz /content/drive/MyDrive/
```

## Expected Interpretation

Compare these rows:

- CDS 4-mer
- CDS NT-v2 meanD/meanG
- TSS 4-mer
- TSS NT-v2 meanD/meanG
- Enformer trunk global/center
- Enformer tracks center

If TSS NT-v2 is also low, then the limitation is mostly genomic context: family5
is primarily a coding-sequence/protein-domain benchmark. If TSS NT-v2 beats
Enformer, then self-supervised DNA representations may preserve broader
sequence information than supervised regulatory output tracks. If Enformer
beats TSS NT-v2, then supervised regulatory pretraining helps on TSS context,
but still does not recover the CDS-level family signal.
