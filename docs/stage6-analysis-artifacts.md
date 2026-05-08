# Stage 6: Build Analysis Artifacts

Stage 6 rebuilds the report-facing analysis tables, figures, and manifest from
tracked cached metrics and feature summaries. It does not rerun expensive
encoder extraction.

## Sample Files

- Input: `samples/stage6_artifact_input.json`
- Output: `samples/stage6_artifact_output.json`

## Full Command

```bash
uv run python scripts/build_analysis_artifacts.py --overwrite
```

Fast smoke run without UMAP:

```bash
uv run python scripts/build_analysis_artifacts.py --out /tmp/dna_analysis_smoke --skip-umap --overwrite
```

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_analysis_artifacts.py` | Builds report-facing tables, figures, and manifest from tracked metrics/caches. |
| `analysis/tables/` | CSV and Markdown tables generated for the report. |
| `analysis/figures/` | PNG figures generated for the report. |
| `analysis/manifest.json` | Manifest listing generated tables and figures. |
| `data/metrics.json` | Source metric log consumed by artifact generation. |
| `data/bootstrap_metrics.json` | Bootstrap CI cache consumed by report summaries. |
| `data/confusion_5way_*.json` | Confusion matrices used for classification figures. |
| `samples/stage6_artifact_input.json` | Tiny summary of artifact builder inputs. |
| `samples/stage6_artifact_output.json` | Tiny summary of generated tables and figures. |

## Inputs

- `data/metrics.json`
- `data/splits.json`
- `data/bootstrap_metrics.json`
- cached probe outputs and confusion matrices in `data/`

## Outputs

- `analysis/tables/` - CSV and Markdown report tables.
- `analysis/figures/` - report figures.
- `analysis/manifest.json` - generated artifact manifest.

The smoke command is the fastest verification that table/figure generation can
still run against the tracked caches.
