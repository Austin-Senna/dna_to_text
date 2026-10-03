# MINA: data, feature caches and stored predictions

This record holds the files behind the MINA benchmark (MLCB 2026) that the code repository does not:
the input sequences, the per-gene feature caches of every encoder, and the stored test predictions of
every probe cell. The code is at https://github.com/Austin-Senna/dna_to_text; use the tag the paper
names, since these files match its records.

## Files

Every tarball unpacks at the repository root and restores the paths the code reads.

| Tarball | Licence | Contents |
|---|---|---|
| `mina_inputs.tar.gz` | CC BY 4.0 | CDS sequences (Ensembl 115), the gene table (family labels, GenePT summaries and targets), the HGNC snapshot, and the stamped feature parquets git does not track (TSS-anchored and TSS composition features) |
| `mina_inputs_nt_v2.tar.gz` | CC BY-NC-SA 4.0 | NT-v2's TSS-anchored parquet (kept apart for its licence) |
| `mina_tss_windows_e115.tar.gz` | CC BY 4.0 | Strand-aware 196,608 bp windows centred on each canonical TSS (Ensembl 115) |
| `mina_predictions_v2.tar.gz` | CC BY 4.0 | Stored test predictions of every canonical probe cell, and the GenePT targets they score against |
| `mina_predictions_v2_nt_v2.tar.gz` | CC BY-NC-SA 4.0 | The same, for the cells on NT-v2 features |
| `mina_cds_chunks_<encoder>.tar.gz` | CC BY 4.0 | DNABERT-2, GENA-LM, HyenaDNA: per-chunk reductions over each CDS |
| `mina_tss_chunks_<encoder>.tar.gz` | CC BY 4.0 | The same three encoders, over each TSS window |
| `mina_enformer_v2.tar.gz` | CC BY 4.0 | Enformer trunk and track features per gene |
| `mina_esm2_v2.tar.gz` | CC BY 4.0 | ESM-2 150M and 650M protein embeddings per gene (fp32) |
| `mina_nt_v2.tar.gz` | CC BY-NC-SA 4.0 | NT-v2 per-chunk reductions over each CDS and TSS window |
| `deposit_manifest.json` | CC BY 4.0 | Every file: its tarball, licence, size and sha256; the commit that packed it and the commit of the records |
| `README.md`, `LICENSES.md` | | This file, and the map from paths to licences |
| `SHA256SUMS` | | Checksums of every file above |

Most pooled feature parquets ship in the repository; the deposit adds only the few git does not track.

## Reproducing

```bash
git clone https://github.com/Austin-Senna/dna_to_text && cd dna_to_text
git checkout <tag>
for t in /path/to/deposit/*.tar.gz; do tar -xzf "$t"; done
uv sync --frozen --python 3.13.13
```

Where to start depends on what you want to check:

- **Statistics from the stored predictions** (no refitting): the predictions tarballs, then
  `uv run scripts/build_statistics.py`. Every prediction file is checked against the hash its record stamps.
- **Every probe cell (Stage 5)**: the two `mina_inputs*` tarballs are enough, then `scripts/recompute_all.sh`. It runs
  on CPU at one thread (about 1 h 40 min across 60 shards on a c6a.16xlarge) and ends with an independent
  reimplementation of the headline cells.
- **Pooling (Stage 4)**: add the cache tarballs. `scripts/check_deposit.sh <deposit dir>` (after `uv sync`;
  it uses the repository's `.venv`) checks every stamped file in a clean copy of the repository, rebuilds
  every stamped parquet from the caches, and checks them again. The composition features need the
  encoders' tokenizers, which it fetches from Hugging Face at the pinned revisions.
- **Extraction (Stages 1 to 3)** needs a GPU. Ours ran on an AWS g5.xlarge (NVIDIA A10G, driver 595.91.07,
  torch 2.11.0+cu130, CUDA 13.0, Python 3.13.13). Model revisions are pinned in
  `src/data_loader/model_registry.py`, and every cache file records its model, revision, input sequence
  hash, card and torch build, so the loaders refuse a file built from anything else.

## Sources

- Ensembl release 115 (GRCh38): CDS, canonical transcripts and the windows.
- HGNC complete set, downloaded 8 April 2026 from the HGNC public download files (CC0).
- GenePT embeddings: Zenodo record 10833191, version 2 (CC BY 4.0).
- MMseqs2 at commit `18cc7493f392b95699ded8c0534dad1558ccc0f1` (https://github.com/soedinglab/MMseqs2) for the
  homology clusters and the purge pairs. Nothing from MMseqs2 is redistributed here.

## Licences

Everything is CC BY 4.0 except the files derived from NT-v2 outputs (`*_nt_v2*` tarballs), which carry
NT-v2's licence, CC BY-NC-SA 4.0. Attribution for the models whose outputs these are:

- DNABERT-2 (Zhou et al., 2024), Apache 2.0.
- NT-v2 (Dalla-Torre et al., 2024), CC BY-NC-SA 4.0.
- GENA-LM (Fishman et al., 2025), MIT (code repository).
- HyenaDNA (Nguyen et al., 2023), BSD 3-Clause.
- Enformer (Avsec et al., 2021), CC BY 4.0, via the EleutherAI PyTorch port.
- ESM-2 (Lin et al., 2023), MIT.

`LICENSES.md` in the repository maps each path to its licence.
