# Stage 4.1: Derive TSS Windows

Stage 4.1 builds one genomic window per gene, centred on the transcription
start site (TSS), for the same gene set used by the CDS experiments. It is
computed from pinned Ensembl files, not downloaded as a dataset.

## Sample Files

- Input: `samples/stage4_1_tss_windows_input.json`
- Output: `samples/stage4_1_tss_windows_output.json`

## Full Commands

```bash
# Ensembl release 115 files into data/annotation/ (gitignored)
B=https://ftp.ensembl.org/pub/release-115
curl -o data/annotation/Homo_sapiens.GRCh38.115.gtf.gz $B/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz
curl -o data/annotation/Homo_sapiens.GRCh38.cdna.all.115.fa.gz $B/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz
curl -o data/annotation/Homo_sapiens.GRCh38.dna.primary_assembly.115.fa.gz $B/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip -k data/annotation/Homo_sapiens.GRCh38.dna.primary_assembly.115.fa.gz

uv run python scripts/build_tss_windows.py           # windows + data/tss_windows.tsv
uv run python scripts/make_tss_disjoint_split.py     # data/splits_tss_disjoint.json
uv run python scripts/run_enformer_features.py --skip-model   # matched TSS 4-mer table
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_tss_windows.py` | Builds every window, checks it, and writes the manifest. |
| `src/data_loader/enformer_windows.py` | Window geometry (`window_span`, `orient`), the local FASTA reader, and `read_window`, the only way consumers read a window. |
| `data/tss_windows.tsv` | Tracked manifest: per gene the transcript, strand, TSS, genomic span, N padding, cDNA prefix and sha256 of the window. |
| `data/tss_windows.meta.json` | Release, window length and the sha256 of each Ensembl input file. |
| `data/tss_windows_e115/` | Ignored window cache, one FASTA per gene. |
| `scripts/make_tss_disjoint_split.py` | Builds the TSS-primary split, disjoint on windows and protein clusters. |
| `src/splits/tss_disjoint.py`, `src/splits/window_leak.py` | Split assignment and the cross-split window-overlap statistics. |
| `analysis/tss_overlap/window_leak.json` | Cross-split window overlap for the homology split and the disjoint split. |
| `scripts/run_enformer_features.py` | Writes matched TSS 4-mer features (and, without `--skip-model`, Enformer features). |
| `data/dataset_enformer_tss_4mer.parquet` | Probe-ready TSS-window 4-mer feature table. |

## Window Definition

- **TSS:** the 5' end of the canonical transcript whose CDS the encoders embed
  (the versioned ENST in `data/sequences/{gene}.fa`), from the Ensembl 115
  GTF. That is the transcript start on the plus strand and the transcript end
  on the minus strand. Release 115 is the one the CDS sequences came from: it
  holds all 3,244 transcript versions, and the builder refuses a release that
  lacks any.
- **Window:** 196,608 bp in gene orientation, with the TSS at index 98,304 (0-based):
  98,304 bases upstream, then the TSS and 98,303 bases downstream.
  Minus-strand windows are reverse-complemented.
- **Chromosome edges:** a window that runs off a chromosome is padded with N on
  that side and never shifted, so the TSS index is the same for every gene. 8 genes are padded,
  from 498 bp (ZBTB45) to 54,023 bp (MZF1). Every encoder sees the padding as input: tokenizers
  that encode N one base at a time turn it into extra chunks, so a heavily padded gene's pooled
  vector is weighted toward the N run.
- **Sequence:** the GRCh38 primary assembly FASTA of the same release.

## Checks

The manifest is written only if both checks pass:

- **cDNA prefix:** the window read from the TSS index matches the first bases
  of the transcript's own cDNA (up to 50, or the first exon if shorter), taken
  from the release's cDNA FASTA, a source independent of the genome. The prefix
  is stored in the manifest so the slow test re-checks every cached window.
- **Regression:** wherever a window's genomic span overlaps the May 2026 window
  (`data/enformer_windows/`, fetched from Ensembl REST), the bases are identical.

## Outputs

- `data/tss_windows.tsv`, `data/tss_windows.meta.json`: tracked manifest.
- `data/tss_windows_e115/{ENSG...}.fa`: ignored window cache.
- `data/splits_tss_disjoint.json`: TSS-primary split, stamped with the sha256 of its inputs.
- `analysis/tss_overlap/window_leak.json`: cross-split window overlap statistics.
- `data/dataset_enformer_tss_4mer.parquet`: matched TSS-window 4-mer baseline table.
