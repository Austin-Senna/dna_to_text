# Courseworks Submission Notes

Deadline: May 7, 2026 at 11:59 PM.

Final report artifact: `dna_to_text.pdf`

Project repository: https://github.com/Austin-Senna/dna_to_text

The report is written as a Bioinformatics Application Note and is intended to be self-contained for a reader with general computational biology background but no prior knowledge of this project.

## Requirement Map

| Courseworks requirement | Repository location |
|---|---|
| Final report PDF | `dna_to_text.pdf` |
| Source code packages | `src/` |
| Experiment and artifact scripts | `scripts/` |
| Code used to generate report display items | `scripts/build_analysis_artifacts.py`, `analysis/viz/` |
| Report figures | `analysis/figures/` |
| Report tables | `analysis/tables/` |
| Bootstrap confidence interval artifact | `data/bootstrap_metrics.json` |
| Small sample inputs and outputs | `samples/` |
| Tests | `tests/` |
| Setup and system requirements | `README.md` |

## Pipeline Sample Inputs and Outputs

The `samples/` directory contains small examples for each pipeline stage:

- Stage 1 curation/fetching: `samples/stage1_curated_genes_input.csv` -> `samples/stage1_curated_genes_output.csv`
- Stage 2 encoder/pooling: `samples/stage2_cds_input.fasta` -> `samples/stage2_encoder_output.json`
- Stage 3 probes/baselines: `samples/stage3_probe_input.csv` -> `samples/stage3_probe_output.json`
- Stage 4.1 TSS-window derivation: `samples/stage4_1_tss_windows_input.json` -> `samples/stage4_1_tss_windows_output.json`
- Stage 4.2 TSS encoders/context ablation: `samples/stage4_2_tss_encoder_input.json` -> `samples/stage4_2_tss_encoder_output.json`
- Stage 5 1000-run bootstrap confidence intervals: `samples/stage5_bootstrap_input.json` -> `samples/stage5_bootstrap_output.json`
- Stage 6 report artifact generation: `samples/stage6_artifact_input.json` -> `samples/stage6_artifact_output.json`

These files are intentionally tiny and reviewer-readable. They show the shape of data entering and leaving each stage; full report reproduction uses the tracked cached artifacts listed below.

## Regenerating Report Artifacts

Regenerate headline bootstrap confidence intervals from tracked cached metrics and feature datasets:

```bash
uv run python scripts/bootstrap_test_uncertainty.py
```

See `docs/stage5-bootstrap.md` for the 1000-run CI protocol and output schema.

Regenerate report tables and figures from tracked cached metrics:

```bash
uv run python scripts/build_analysis_artifacts.py --overwrite
```

Fast artifact smoke run without UMAP:

```bash
uv run python scripts/build_analysis_artifacts.py --out /tmp/dna_analysis_smoke --skip-umap --overwrite
```

These commands regenerate the analysis artifacts used to support the report. They do not rerun the expensive encoder extraction jobs.

## Public Large Data and Model Sources

- GenePT v2 artifacts: Zenodo DOI `10.5281/zenodo.10833191`; unzip `GenePT_emebdding_v2.zip` into `GenePT_emebdding_v2/`.
- HGNC complete gene set: `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt`.
- Ensembl canonical CDS: Ensembl REST `/lookup/id/{gene_id}` and `/sequence/id/{transcript_id}?type=cds`.
- Ensembl TSS windows: derived from Ensembl REST gene coordinates via `src/data_loader/enformer_windows.py`, then fetched from `/sequence/region/human/{region}` as 196,608 bp strand-aware TSS-centered windows.
- DNABERT-2 checkpoint: `zhihan1996/DNABERT-2-117M`.
- Nucleotide Transformer checkpoint: `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`.
- GENA-LM checkpoint: `AIRI-Institute/gena-lm-bert-base-t2t`.
- HyenaDNA checkpoint: `LongSafari/hyenadna-large-1m-seqlen-hf`.
- Optional Enformer comparator dependency: install with `uv pip install ".[enformer]"`.

## Full-Pipeline Caveat

The full project operates on public genomic resources and pretrained model checkpoints, but the encoder extraction stages are computationally expensive. The repository therefore tracks the small report-supporting caches needed to reproduce tables, figures, bootstrap intervals, and sample pipeline outputs without rerunning every encoder.
