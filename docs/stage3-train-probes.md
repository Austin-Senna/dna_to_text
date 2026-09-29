# Stage 3: Train Probes and Baselines

Stage 3 freezes the train/validation/test split, trains family-classification
and Ridge-to-GenePT probes, and builds the main metric tables.

## Sample Files

- Input: `samples/stage3_probe_input.csv`
- Output: `samples/stage3_probe_output.json`

## Full Commands

Probe fits refuse to run unless every BLAS/OpenMP pool is pinned to one thread
(thread count changes lbfgs results), so set the pin before Python starts:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python scripts/make_splits.py
uv run python scripts/train_logistic_probe.py --dataset nt_v2_meanD --task family5
uv run python scripts/train_probe.py --dataset data/dataset_dnabert2_meanG.parquet
uv run python scripts/build_family5_table.py
uv run python scripts/build_regression_table.py
```

Additional encoder/pooling cells use the same probe scripts with different
`--dataset` values; `--splits` names the split file (default `data/splits.json`).

Every probe CLI runs one cell through `linear_trainer.cell.run_cell` under the V2
protocol (`linear_trainer.protocol.V2`): StandardScaler inside the fit, float64,
C or alpha on decades from 1e-4 to 1e4, extended one decade at a time while the
validation pick sits on an edge (a pick left on an edge is flagged in the record),
lbfgs `tol=1e-6`, `max_iter=5000`. The test split is loaded only after selection
and the refit, the test predictions are stored for the bootstraps, and each record
carries a provenance stamp (git SHA, split file and its sha256, protocol hash,
thread count, `n_iter`).

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/make_splits.py` | Creates the frozen 70/15/15 train/validation/test split. |
| `scripts/check_split_reproduction.py` | Re-clusters with the local MMseqs2 build and checks that it reproduces `data/splits.json` (40%) or `data/splits_homology70.json` (70%); `--keep-tsv` saves the cluster assignments. |
| `scripts/train_logistic_probe.py` | Trains multinomial family5 probes and classification baselines. |
| `scripts/train_probe.py` | Trains Ridge probes from DNA features into GenePT text embeddings. |
| `scripts/train_baseline.py` | Runs 4-mer Ridge baseline cells. |
| `scripts/train_anti_baseline.py` | Runs shuffled-GenePT anti-baseline cells for leakage checks. |
| `scripts/build_family5_table.py` | Builds the main family-classification summary table. |
| `scripts/build_regression_table.py` | Builds the main Ridge-to-GenePT summary table. |
| `src/splits/make_splits.py` | Split construction helpers used by the split CLI. |
| `src/splits/loader.py` | Loads split-specific `X`, `Y`, and metadata arrays from feature tables. |
| `src/linear_trainer/protocol.py` | The probe protocol (`V2`), the thread-pin assert and the provenance stamp. |
| `src/linear_trainer/fit.py` | The single fit path: fit, validation sweep with grid extension and edge flags, test metrics. |
| `src/linear_trainer/cell.py` | One cell end to end (`run_cell`), stored test predictions, `load_predictions`. |
| `src/linear_trainer/sources.py` | Feature-source registry: encoder parquets, comparators and on-the-fly composition features. |
| `src/linear_trainer/selection.py` | Validation-based selection over recorded runs, with explicit pool candidates. |
| `src/linear_trainer/logistic_probe.py` | Fitted logistic probe (weights plus the scaler it was fitted with). |
| `src/linear_trainer/probe.py` | Fitted Ridge probe: prediction and serialization. |
| `src/kmer_baseline/featurizer.py` | 4-mer composition feature extraction. |
| `samples/stage3_probe_input.csv` | Tiny example of feature rows entering probe training. |
| `samples/stage3_probe_output.json` | Tiny example of probe metrics and selected hyperparameters. |

## Outputs

- `data/splits.json` - frozen 70/15/15 split.
- `data/clusters/homology_id40.tsv`, `data/clusters/homology_id70.tsv` - canonical MMseqs2 cluster assignments (representative, member) behind the two homology splits.
- `data/metrics.json` - appended probe and baseline metrics.
- `data/confusion_5way_*.json` - family-classification confusion summaries from the May 2026 runs. Probe runs now write them only for unshuffled family5 cells, and only into the directory given by `--confusion-dir` (pass `--confusion-dir data` to refresh these).
- `outputs/predictions/<metrics stem>/*.npz` - each cell's stored test predictions (content-addressed names), which the bootstraps rescore; `--pred-dir` overrides the location (gitignored).
- `analysis/tables/main_family5.md` - best family5 cell per encoder.
- `analysis/tables/main_regression.md` - best Ridge-to-GenePT cell per encoder.

## Headline Results

- Best family classification: NT-v2 `meanD`, macro-F1 0.8275 and kappa 0.8214.
- CDS 4-mer family baseline: macro-F1 0.6722 and kappa 0.7024.
- Best GenePT regression: DNABERT-2 `meanG`, macro R2 0.2104.
- CDS 4-mer regression baseline: macro R2 0.1743.
