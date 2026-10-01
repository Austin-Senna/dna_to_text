# Stage 7: Build Paper Figures and Tables

Stage 7 renders the report-facing figures and LaTeX table fragments that the
manuscript (`dna_to_text_paper/`) `\includegraphics` / `\input`s, from the
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
uv run scripts/build_paper_tables.py     # LaTeX table fragments -> dna_to_text_paper/paper/tables/
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_result_figures.py` | Results figures: comparator macro-F1 / GenePT R^2 (3.1/3.2), encoder x pooling heatmap (3.3), CDS-vs-TSS substrate ablation (3.4), random-vs-homology split (3.5). `comparator_f1_bands.png` is a candidate macro-F1 panel that draws the ESM-2 650M shuffled-label band beside the 4-mer's. |
| `scripts/build_umap_compare.py` | UMAP of the validation-selected NT-v2 CDS and TSS features beside ESM-2. |
| `scripts/build_statistics.py` | Cluster-bootstrap intervals, paired tests (Holm over the four confirmatory tests), null bands; requires `data/v2/run_complete.json` and a complete, passing `data/v2/reproduction.json` for the same records, and cross-checks T1-T4 against it. |
| `scripts/build_counts.py` | The gene counts the text states, each with its denominator: single-chunk share per encoder (from the v2 chunk caches), noisy TF and kinase labels and templated GenePT summaries (`src/data_loader/label_audit.py`). |
| `scripts/ridge_robust_metrics.py` | Rescores stored GenePT predictions: pooled R^2 and retrieval; control row = the 200-shuffle null band. |
| `scripts/build_paper_tables.py` | LaTeX table-body fragments: best-cell classification/regression, substrate ablation, split comparison (random against each arm's primary split: homology for CDS, disjoint for TSS), seed sensitivity, intervals, paired differences, the D5 sensitivity table (`s_d5_sensitivity.tex`: Ends + Mean against Mean at 3x C, the best encoder against composition plus CDS length, and the headline tests without noisy TF labels, without non-protein-kinase kinase labels, or without templated GenePT targets), the test-population profile (`s_split_population.tex`, from `data/v2/counts.json`: per partition, singleton share, median 40% cluster size, share in clusters of 10 or more, olfactory receptors among GPCRs), and the appendix matrices. |
| `scripts/build_poster_figures.py` | Poster PDFs into `poster/figures/` from the frozen May accessors in `scripts/poster_may_records.py` (not for the paper). GENA-LM's CDS cells come from `data/metrics_poster_gena_lm.json` (rerun after the weight-loading fix), and the TSS panel uses the disjoint split. |

## Inputs

- `data/v2/metrics_<split stem>.json`, `data/v2/null_<split stem>.json` - records from `scripts/recompute_all.sh` (one commit and protocol; the loader refuses mixtures).
- `data/v2/run_complete.json` - written by the final unsharded run (completeness and the G1 check).
- `data/v2/reproduction.json` - written by `scripts/reproduce_headline.py` (the independent reimplementation), which `scripts/recompute_all.sh all` runs at the end of an unsharded pass; `build_statistics.py` refuses records it did not check.
- `data/v2/counts.json` - from `scripts/build_counts.py`; `build_paper_tables.py` reads its test-population profile and refuses one built from other split files.
- `data/v2/statistics.json`, `data/v2/ridge_robust.json` - each stamped with the records it was built from; the builders refuse a stale one.
- Stored test predictions in `outputs/predictions/v2/` (not tracked).

## Outputs

- `dna_to_text_paper/paper/figures/*.png` - Results figures (no embedded titles; the LaTeX caption is the title, per repo convention).
- `dna_to_text_paper/paper/tables/*.tex` - table-body fragments the paper `\input`s.

The builders are deterministic given the `data/v2` records and statistics of one
commit: re-running reproduces the same figures and fragments.
