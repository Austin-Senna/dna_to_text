# Stage 1: Curate Genes and Fetch CDS

Stage 1 builds the gene table used by every downstream experiment. It joins
GenePT symbols to HGNC metadata, assigns each gene to one functional family,
fetches canonical CDS sequence from Ensembl, and writes the cached project
corpus.

## Sample Files

- Input: `samples/stage1_curated_genes_input.csv`
- Output: `samples/stage1_curated_genes_output.csv`

## Full Command

```bash
uv run python scripts/prepare_data.py
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/prepare_data.py` | CLI entrypoint for building the curated gene table and fetching CDS sequences. |
| `src/data_loader/dataset_loader.py` | Loads GenePT and HGNC, defines family regex rules, joins symbols, and assigns each gene to one family. |
| `src/data_loader/sequence_fetcher.py` | Fetches and caches canonical CDS FASTA from Ensembl REST. |
| `src/data_loader/pipeline.md` | Older detailed note on the HGNC/GenePT/Ensembl join and cache layout. |
| `samples/stage1_curated_genes_input.csv` | Tiny example of starting gene/family requests. |
| `samples/stage1_curated_genes_output.csv` | Tiny example of curated rows after HGNC/GenePT/Ensembl resolution. |
| `data/gene_table.parquet` | Full curated corpus produced by this stage. |
| `data/sequences/` | Ignored local CDS FASTA cache used by later stages. |

## Inputs

- GenePT v2 artifacts from Zenodo DOI `10.5281/zenodo.10833191`.
- HGNC complete gene set from `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt`.
- Ensembl REST canonical CDS endpoint `/sequence/id/{transcript_id}?type=cds`.

## Outputs

- `data/gene_table.parquet` - curated metadata, family label, GenePT target, and Ensembl ID.
- `data/sequences/{ENSG...}.fa` - cached canonical CDS FASTA files.

Large source artifacts and sequence caches are generated locally and ignored by
git. The tracked sample files show the row shape expected before and after this
stage.
