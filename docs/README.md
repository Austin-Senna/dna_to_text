# Documentation

This directory is split into stage-level pipeline notes and archived planning
history. Active docs mirror the sample input/output stages in `../samples/`.

## Stage Docs

- `stage1-curate-genes.md` - curate genes, join GenePT/HGNC, and fetch CDS.
- `stage2-encode-cds.md` - run CDS encoders and build pooling datasets.
- `stage3-train-probes.md` - train family5 and Ridge-to-GenePT probes.
- `stage4-1-tss-windows.md` - derive strand-aware TSS-centered windows.
- `stage4-2-tss-encoders.md` - run TSS NT-v2 and Enformer context ablations.
- `stage5-bootstrap.md` - 1,000-run bootstrap confidence interval protocol.
- `stage6-analysis-artifacts.md` - regenerate report tables and figures.

## Submission Entry Points

- `../README.md` - repository layout, pipeline, setup, testing, and troubleshooting.
- `../submission.md` - Courseworks requirement map and submission notes.
- `../samples/README.md` - small input/output examples for each pipeline stage.

## Archive

- `archive/project-history/` - original pitch, framework, next-step log, and writeup scaffold.
- `archive/notes/` - older method notes and paper-revision follow-up log.

Generated report tables and figures live outside this docs tree in
`analysis/tables/` and `analysis/figures/`.
