# Stage 3: Train Probes and Baselines

Stage 3 freezes the train/validation/test split, trains family-classification
and Ridge-to-GenePT probes, and builds the main metric tables.

## Sample Files

- Input: `samples/stage3_probe_input.csv`
- Output: `samples/stage3_probe_output.json`

## Full Commands

```bash
uv run python scripts/make_splits.py
uv run python scripts/train_logistic_probe.py --dataset nt_v2_meanD --task family5
uv run python scripts/train_probe.py --dataset data/dataset_dnabert2_meanG.parquet
uv run python scripts/build_family5_table.py
uv run python scripts/build_regression_table.py
```

Additional encoder/pooling cells use the same probe scripts with different
`--dataset` values.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/make_splits.py` | Creates the frozen 70/15/15 train/validation/test split. |
| `scripts/train_logistic_probe.py` | Trains multinomial family5 probes and classification baselines. |
| `scripts/train_probe.py` | Trains Ridge probes from DNA features into GenePT text embeddings. |
| `scripts/train_baseline.py` | Runs 4-mer Ridge baseline cells. |
| `scripts/train_anti_baseline.py` | Runs shuffled-GenePT anti-baseline cells for leakage checks. |
| `scripts/build_family5_table.py` | Builds the main family-classification summary table. |
| `scripts/build_regression_table.py` | Builds the main Ridge-to-GenePT summary table. |
| `src/splits/make_splits.py` | Split construction helpers used by the split CLI. |
| `src/splits/loader.py` | Loads split-specific `X`, `Y`, and metadata arrays from feature tables. |
| `src/linear_trainer/logistic_probe.py` | Logistic probe fitting and hyperparameter sweep helpers. |
| `src/linear_trainer/probe.py` | Ridge probe fitting, prediction, and serialization helpers. |
| `src/kmer_baseline/featurizer.py` | 4-mer composition feature extraction. |
| `samples/stage3_probe_input.csv` | Tiny example of feature rows entering probe training. |
| `samples/stage3_probe_output.json` | Tiny example of probe metrics and selected hyperparameters. |

## Outputs

- `data/splits.json` - frozen 70/15/15 split.
- `data/metrics.json` - appended probe and baseline metrics.
- `data/confusion_5way_*.json` - family-classification confusion summaries.
- `analysis/tables/main_family5.md` - best family5 cell per encoder.
- `analysis/tables/main_regression.md` - best Ridge-to-GenePT cell per encoder.

## Headline Results

- Best family classification: NT-v2 `meanD`, macro-F1 0.8275 and kappa 0.8214.
- CDS 4-mer family baseline: macro-F1 0.6722 and kappa 0.7024.
- Best GenePT regression: DNABERT-2 `meanG`, macro R2 0.2104.
- CDS 4-mer regression baseline: macro R2 0.1743.
