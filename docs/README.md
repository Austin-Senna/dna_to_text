# Documentation

This directory is split into stage-level pipeline notes and archived planning
history. Active docs mirror the sample input/output stages in `../samples/`.

## Stage Docs

- `stage1-curate-genes.md` - curate genes, join GenePT/HGNC, and fetch CDS.
- `stage2-encode-cds.md` - run CDS encoders and build pooling datasets.
- `stage3-train-probes.md` - build the homology-aware splits and the evaluation purge, and train family5 and Ridge-to-GenePT probes under the V2 protocol.
- `stage4-1-tss-windows.md` - derive strand-aware TSS-centered windows and the TSS-primary split (`data/splits_tss_disjoint.json`).
- `stage4-2-tss-encoders.md` - run the four DNA encoders and Enformer on TSS windows, plus the GPU extraction pilot tooling.
- `stage5-bootstrap.md` - cluster-bootstrap intervals, the four primary tests (T1-T4) and the null bands.
- Stage 6 (report analysis artifacts) is retired.
- `stage7-paper-figures-tables.md` - render the manuscript figures and LaTeX table fragments.

## Submission Entry Points

- `../README.md` - repository layout, pipeline, setup, testing, and troubleshooting.
- `../submission.md` - Courseworks requirement map and submission notes.
- `../samples/README.md` - small input/output examples for each pipeline stage.

## Archive

- `archive/project-history/` - original pitch, framework, next-step log, and writeup scaffold.
- `archive/notes/` - older method notes and paper-revision follow-up log.

The paper's tables and figures are generated into the paper submodule
(`dna_to_text_paper/paper/tables/`, `dna_to_text_paper/paper/figures/`); see
`stage7-paper-figures-tables.md`. Stage 6 is retired and its outputs removed.
