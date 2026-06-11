# Stage 7: Build Paper Figures and Tables

Stage 7 renders the report-facing figures and LaTeX table fragments that the
manuscript (`dna_to_text_paper/`) `\includegraphics` / `\input`s, from the
tracked metric JSONs. It writes into the paper submodule and reruns no encoder
extraction or probes.

## Commands

```bash
uv run scripts/build_result_figures.py   # Results figures -> dna_to_text_paper/paper/figures/
uv run scripts/build_paper_tables.py     # LaTeX table fragments -> dna_to_text_paper/paper/tables/
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_result_figures.py` | Results figures: comparator macro-F1 / GenePT R^2 (3.1/3.2), encoder x pooling heatmap (3.3), CDS-vs-TSS substrate ablation (3.4), random-vs-homology split (3.5). |
| `scripts/build_paper_tables.py` | LaTeX table-body fragments: best-cell classification/regression, substrate ablation, split comparison, and the appendix matrices. |
| `data/metrics_homology.json` | Primary (homology-split) metric log consumed by both. |
| `data/metrics.json`, `data/metrics_random_comparators.json` | Random-split values for the leakage / split-comparison artifacts. |
| `data/metrics_enformer_homology.json` | Enformer comparator re-probed on the homology split. |
| `data/bootstrap_metrics.json` | Paired-bootstrap CIs for the appendix table. |

## Inputs

- `data/metrics_homology.json`
- `data/metrics.json`, `data/metrics_random_comparators.json`
- `data/metrics_enformer_homology.json`
- `data/bootstrap_metrics.json`

## Outputs

- `dna_to_text_paper/paper/figures/*.png` - Results figures (no embedded titles; the LaTeX caption is the title, per repo convention).
- `dna_to_text_paper/paper/tables/*.tex` - table-body fragments the paper `\input`s.

Both scripts are idempotent against the tracked metric caches: re-running
reproduces the committed figures and fragments byte-for-byte.
