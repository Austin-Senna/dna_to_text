# Documentation

Stage-level pipeline notes, one per stage of the pipeline described in the top-level `README.md`, plus the Zenodo deposit description. The sample inputs and outputs in `../samples/` follow the same stages.

- `stage1-curate-genes.md` - curate genes, join GenePT/HGNC, and fetch CDS.
- `stage2-encode-cds.md` - run CDS encoders and build pooling datasets.
- `stage3-train-probes.md` - build the homology-aware splits and the evaluation purge, and train family-classification and Ridge-to-GenePT probes.
- `stage4-1-tss-windows.md` - derive strand-aware TSS-centred windows and the interval-disjoint split (`data/splits_tss_disjoint.json`).
- `stage4-2-tss-encoders.md` - run the four DNA encoders and Enformer on TSS windows.
- `stage5-bootstrap.md` - cluster-bootstrap intervals, the four primary tests and the null bands.
- `stage7-paper-figures-tables.md` - render the paper's figures, LaTeX table fragments and prose numbers.
- `zenodo_deposit.md` - what the Zenodo record holds and how to rerun from it.

Stage 6 (report analysis artifacts) is retired. The paper's tables and figures are generated into the paper's LaTeX source tree (`dna_to_text_paper/paper/tables/`, `dna_to_text_paper/paper/figures/`), which is not part of this repository.
