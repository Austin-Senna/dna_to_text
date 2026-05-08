# Stage 4.1: Derive TSS Windows

Stage 4.1 derives TSS-centered genomic windows for the same gene set used by
the CDS experiments. This is not a separate downloaded dataset: it is computed
from Ensembl gene coordinates and fetched from Ensembl REST.

## Sample Files

- Input: `samples/stage4_1_tss_windows_input.json`
- Output: `samples/stage4_1_tss_windows_output.json`

## Full Command

```bash
uv run python scripts/run_enformer_features.py --skip-model
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/run_enformer_features.py` | Fetches TSS windows, writes matched TSS 4-mer features, and optionally runs Enformer. |
| `src/data_loader/enformer_windows.py` | Looks up Ensembl gene coordinates, derives strand-aware TSS windows, and fetches region FASTA. |
| `src/kmer_baseline/featurizer.py` | Converts TSS-window sequence into 4-mer composition features. |
| `samples/stage4_1_tss_windows_input.json` | Tiny example of metadata needed to derive TSS windows. |
| `samples/stage4_1_tss_windows_output.json` | Tiny example of TSS-window cache and matched 4-mer outputs. |
| `data/enformer_windows/` | Ignored local FASTA cache, one TSS-centered window per gene. |
| `data/dataset_enformer_tss_4mer.parquet` | Probe-ready TSS-window 4-mer feature table. |

## Inputs

- Template metadata from `data/dataset_nt_v2_meanD.parquet`.
- Ensembl REST gene coordinates from `/lookup/id/{gene_id}`.
- Ensembl REST genomic sequence from `/sequence/region/human/{region}`.

## Window Definition

For each Ensembl gene ID, `src/data_loader/enformer_windows.py` chooses the
strand-aware TSS coordinate: gene start on the positive strand and gene end on
the negative strand. It then fetches a 196,608 bp human reference-genome window
centered on that coordinate.

## Outputs

- `data/enformer_windows/{ENSG...}.fa` - ignored local TSS-window FASTA cache.
- `data/dataset_enformer_tss_4mer.parquet` - matched TSS-window 4-mer baseline table.
