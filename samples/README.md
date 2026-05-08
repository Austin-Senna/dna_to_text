# Pipeline Sample Inputs and Outputs

These files are small, reviewer-readable examples for each stage of the project pipeline. They are not a replacement for the tracked cached datasets used by the report; they show the shape of data entering and leaving each stage.

## Stage 1: Curate Genes and Fetch CDS

- Input: `stage1_curated_genes_input.csv`
- Output: `stage1_curated_genes_output.csv`
- Full command: `uv run python scripts/prepare_data.py`
- Details: `docs/stage1-curate-genes.md`

## Stage 2: Encode CDS and Pool Features

- Input: `stage2_cds_input.fasta`
- Output: `stage2_encoder_output.json`
- Full commands: `scripts/run_*encoder*.py`, `scripts/run_multi_pool_extract.py`, and `scripts/build_pooling_datasets.py`
- Details: `docs/stage2-encode-cds.md`

## Stage 3: Train Linear Probes and Baselines

- Input: `stage3_probe_input.csv`
- Output: `stage3_probe_output.json`
- Full commands: `scripts/train_logistic_probe.py`, `scripts/train_probe.py`, and baseline/anti-baseline scripts
- Details: `docs/stage3-train-probes.md`

## Stage 4.1: Derive TSS Windows

- Input: `stage4_1_tss_windows_input.json`
- Output: `stage4_1_tss_windows_output.json`
- Full command: `uv run python scripts/run_enformer_features.py --skip-model`
- Details: `docs/stage4-1-tss-windows.md`

## Stage 4.2: Run TSS Encoders and Context Ablation

- Input: `stage4_2_tss_encoder_input.json`
- Output: `stage4_2_tss_encoder_output.json`
- Full commands: `scripts/run_enformer_features.py`, `scripts/run_tss_multi_pool_extract.py`, `scripts/build_tss_pooling_datasets.py`, `scripts/train_logistic_probe.py`, and `scripts/train_probe.py`
- Details: `docs/stage4-2-tss-encoders.md`

## Stage 5: Bootstrap Test-Set Confidence Intervals

- Input: `stage5_bootstrap_input.json`
- Output: `stage5_bootstrap_output.json`
- Full command: `uv run python scripts/bootstrap_test_uncertainty.py`
- Details: `docs/stage5-bootstrap.md`

## Stage 6: Build Report Analysis Artifacts

- Input: `stage6_artifact_input.json`
- Output: `stage6_artifact_output.json`
- Full command: `uv run python scripts/build_analysis_artifacts.py --overwrite`
- Details: `docs/stage6-analysis-artifacts.md`
