# Stage 7: Build Paper Figures and Tables

Stage 7 renders the report-facing figures, LaTeX table fragments and prose numbers
(`paper/numbers.tex`) that the manuscript (`dna_to_text_paper/`) `\includegraphics` / `\input`s, from the
camera-ready records in `data/v2/` (read through `linear_trainer.records`) and
`data/v2/statistics.json`. It writes into the paper submodule and reruns no encoder
extraction or probes. Every pick is made on validation scores only.

## Commands

```bash
# after scripts/recompute_all.sh all (see the repository README)
uv run scripts/build_statistics.py       # intervals, paired tests, null bands -> data/v2/statistics.json
uv run scripts/ridge_robust_metrics.py   # rotation-invariant Ridge metrics -> data/v2/ridge_robust.json
uv run scripts/build_counts.py           # single-chunk shares, noisy TF and kinase labels, templated summaries -> data/v2/counts.json
uv run scripts/build_result_figures.py   # Results figures -> dna_to_text_paper/paper/figures/
uv run scripts/build_umap_compare.py     # UMAP figures -> dna_to_text_paper/paper/figures/
uv run scripts/build_selection_sensitive.py  # the moved cells under both perturbations -> data/v2/selection_sensitive.json
uv run scripts/build_paper_tables.py     # LaTeX table fragments -> dna_to_text_paper/paper/tables/
uv run scripts/build_numbers.py          # every prose number -> dna_to_text_paper/paper/numbers.tex
uv run scripts/build_numbers.py --check  # submission gate: no \pending, undefined \val, or bare result decimal
```

The prose never types a result: it prints `\val{key}` from `numbers.tex` (key families in
`scripts/build_numbers.py`'s docstring). A printed digit that moves when the frozen protocol is
refitted at 6 threads or on another BLAS kernel carries a dagger (`\sens{}`, defined in
`paper/header.tex`), in the tables and in the prose alike.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_result_figures.py` | Results figures: comparator macro-F1 / GenePT R^2 (3.1/3.2), encoder x pooling heatmap (3.3), CDS-vs-TSS substrate ablation (3.4), random-vs-homology split (3.5). `comparator_f1_bands.png` (the 4-mer and ESM-2 650M shuffled-label bands) is the paper's Figure 2. |
| `scripts/build_umap_compare.py` | UMAP of the validation-selected NT-v2 CDS and TSS features (Appendix Figure A1). |
| `scripts/build_statistics.py` | Cluster-bootstrap intervals, paired tests (Holm over the four confirmatory tests), null bands; requires `data/v2/run_complete.json` and a complete, passing `data/v2/reproduction.json` for the same records, and cross-checks T1-T4 against it. |
| `scripts/build_counts.py` | The gene counts the text states, each with its denominator: single-chunk share per encoder (from the v2 chunk caches), noisy TF and kinase labels and templated GenePT summaries (`src/data_loader/label_audit.py`). |
| `scripts/ridge_robust_metrics.py` | Rescores stored GenePT predictions: pooled R^2 and retrieval; control row = the 200-shuffle null band. |
| `scripts/build_paper_tables.py` | LaTeX table-body fragments: best-cell classification/regression, substrate ablation, split comparison (random against each arm's primary split: homology for CDS, disjoint for TSS), seed sensitivity, intervals, paired differences, the D5 sensitivity table (`s_d5_sensitivity.tex`: Ends + Mean against Mean at 3x C, the best encoder against composition plus CDS length, and the headline tests without noisy TF labels, without non-protein-kinase kinase labels, or without templated GenePT targets), the test-population profile (`s_split_population.tex`, from `data/v2/counts.json`: per partition, singleton share, median 40% cluster size, share in clusters of 10 or more, olfactory receptors among GPCRs), and the appendix matrices. Each fragment is built from the canonical records and again with each perturbation in `selection_sensitive.json` replayed; a number whose text differs carries `\sens{}`, and a replay the tables cannot carry (a flipped primary-split pick, a moved cell behind `ridge_robust.json`) refuses to build. |
| `scripts/build_selection_sensitive.py` | Merges the two `diff_records.py` reports into the moved-cell list, with every metric delta and the sha256 of the canonical records it applies to. |
| `scripts/build_numbers.py` | Writes `paper/numbers.tex`: every number the prose states, under a key named for what it measures, replayed through both perturbations like the tables. Computes the homology split's TSS-window overlap itself from the current split and window manifest. Reads the TSS-window composition from `scripts/tss_overlap.py`'s audit table and refuses one built from another window manifest, GTF or gene set. `--check` is the submission gate. Tests: `tests/test_build_numbers.py`, `tests/test_selection_marks.py`. |
| `scripts/build_poster_figures.py` | Poster PDFs into `poster/figures/` from the frozen May accessors in `scripts/poster_may_records.py` (not for the paper). GENA-LM's CDS cells come from `data/metrics_poster_gena_lm.json` (rerun after the weight-loading fix), and the TSS panel uses the disjoint split. |

## Inputs

- `data/v2/metrics_<split stem>.json`, `data/v2/null_<split stem>.json` - records from `scripts/recompute_all.sh` (one commit and protocol; the loader refuses mixtures).
- `data/v2/run_complete.json` - written by the final unsharded run (completeness and the G1 check).
- `data/v2/reproduction.json` - written by `scripts/reproduce_headline.py` (the independent reimplementation), which `scripts/recompute_all.sh all` runs at the end of an unsharded pass; `build_statistics.py` refuses records it did not check.
- `data/v2/counts.json` - from `scripts/build_counts.py`; `build_paper_tables.py` reads its test-population profile and refuses one built from other split files.
- `data/v2/statistics.json`, `data/v2/ridge_robust.json` - each stamped with the records it was built from; the builders refuse a stale one.
- Stored test predictions in `outputs/predictions/v2/` (not tracked).
- `data/gene_table.parquet` and `data/hgnc/` (not tracked) - the Stage 1 gene table and HGNC groups, read by `build_statistics.py` and `build_counts.py`.
- `data/v2/selection_sensitive.json` - from `scripts/build_selection_sensitive.py` over `data/v2/determinism_t6.json` (the main and null groups refitted at 6 threads) and `data/v2/determinism_kernel.json` (the clean room on another OpenBLAS kernel), both `scripts/diff_records.py` reports: the cells whose pick or scores move under either perturbation, with their metric deltas, which the tables and `numbers.tex` replay to mark moving digits.
- `data/tss_windows.tsv` and the split files - `build_numbers.py` computes the TSS-window overlap from them (it must be the manifest the records' window purge read).
- `analysis/tss_overlap/tables/overlap_by_family.csv` and `provenance.json` - from `scripts/tss_overlap.py` (tracked); checked against `data/tss_windows.tsv` and the GTF pin in `data/tss_windows.meta.json`.

## Outputs

- `dna_to_text_paper/paper/figures/*.png` - Results figures (no embedded titles; the LaTeX caption is the title, per repo convention).
- `dna_to_text_paper/paper/tables/*.tex` - table-body fragments the paper `\input`s.
- `dna_to_text_paper/paper/numbers.tex` - the prose numbers (`\val{key}`).

The builders are deterministic given the `data/v2` records and statistics of one
commit: re-running reproduces the same figures, fragments and `numbers.tex`.
