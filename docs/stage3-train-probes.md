# Stage 3: Train Probes and Baselines

Stage 3 builds the train/validation/test splits and trains family-classification
and Ridge-to-GenePT probes. The paper's tables are built in Stage 7.

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
```

Additional encoder/pooling cells use the same probe scripts with different
`--dataset` values; `--splits` names the split file (default `data/splits.json`).
The CLIs do not choose a split from the source, so a TSS source needs
`--splits data/splits_tss_disjoint.json` to run on its primary split.

## Splits

`scripts/make_splits.py` writes three split files, all at seed 42 with a 70/15/15
target per family:

- `data/splits.json`, the CDS primary split. Each gene's cached CDS is translated
  up to the first stop codon, and the proteins are clustered with
  `mmseqs easy-cluster` at `--min-seq-id 0.4`, `-c 0.8` and `--cov-mode 0`
  (coverage of both sequences; `src/cluster/mmseqs_cluster.py`). Whole clusters
  are then assigned greedily, largest first, to the split whose per-family quota
  they overshoot least (`splits.make_splits.build_cluster_splits`), so no cluster
  spans two splits.
- `data/splits_homology70.json`, the same construction at 70% identity.
- `data/splits_random.json`, a random family-stratified split. It is the leakage
  demonstration and is not purged.

Clustering alone does not separate every homologue: two genes in different
clusters can still be close. Every cell therefore applies an evaluation purge
(`splits.leaks.purge_for`, policy in `splits.leaks.rules_for`). The pair table
`data/leaks/protein_pairs.tsv` (`scripts/build_protein_pairs.py`) comes from an
all-vs-all MMseqs2 search over the full-length proteins and keeps every pair that
passes Rule A: E <= 1e-3, identity >= 0.4, coverage >= 0.8 on both sequences.
Pairs inside one split are ignored. A validation gene with a partner in train is
masked from selection, and a test gene with a partner in train or validation is
masked from scoring. Training is never masked, so the fits do not change. The
homology splits and their seeds use Rule A at 40%, the 70% split Rule A at 70%,
and the TSS-disjoint splits Rule A plus window overlap (Stage 4.1). A split file
that `rules_for` does not name raises instead of running unpurged. The record
carries a `purge` field and the unpurged test metrics for disclosure.

The TSS cells have their own primary split, `data/splits_tss_disjoint.json`
(Stage 4.1); for them, `data/splits.json` is a sensitivity analysis.

## Probe Protocol

Every probe CLI runs one cell through `linear_trainer.cell.run_cell` under the V2
protocol (`linear_trainer.protocol.V2`): StandardScaler inside the fit, float64,
C or alpha on decades from 1e-4 to 1e4, extended one decade at a time while the
validation pick sits on an edge (a pick left on an edge is flagged in the record),
lbfgs `tol=1e-6`, `max_iter=5000`. The test split is loaded only after selection
and the refit, the test predictions are stored for the bootstraps, and each record
carries a provenance stamp (git SHA, split file and its sha256, protocol hash,
thread count, `n_iter`). `protocol.assert_threads` checks the thread pin before
every cell. V2 replaced the May protocol (no scaler, float32, a 6-point grid, the
scikit-learn default `tol`, `max_iter=2000`, the machine's default thread count;
`LEGACY` in `tests/legacy_cell.py`), so May numbers are not comparable with V2
records.

One control source is derived at load time rather than read from its own parquet:
`<encoder>_meanmean3` is the Mean pool copied three times (`sources.DERIVED`). Under
StandardScaler and an L2 penalty it is exactly Mean at 3x C (Ridge: alpha / 3), and it
equals Ends + Mean for every single-chunk gene, so Ends + Mean against it isolates the
chunk-position effect. It runs on the CDS primary split only and is never a pool
candidate. The `<k-mer>_len` sources (`sources.LENGTH_CONTROLS`) are another control on
the CDS primary only: a composition vector plus log1p(CDS length), since the k-mer
vectors are L1-normalised while the encoders see length through their chunking. They
are never a k-mer pick.

The camera-ready run is one manifest, not per-cell commands: `scripts/recompute_all.sh all`
runs every cell and null band into `data/v2/` (see the repository README), then
`scripts/reproduce_headline.py` refits the headline cells independently and checks them
against the records (docs/stage5-bootstrap.md). `--threads N` refits at N threads into a trial
directory only (refused for `data/v2`), and `scripts/diff_records.py` compares two runs cell by cell;
together they give the determinism checks that `scripts/build_selection_sensitive.py` merges into
`data/v2/selection_sensitive.json`.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/make_splits.py` | Writes the homology splits `data/splits.json` (40%) and `data/splits_homology70.json` (70%) and the random split `data/splits_random.json`; refuses any seed but 42 (seed splits come from `make_seed_splits.py`). |
| `scripts/check_split_reproduction.py` | Re-clusters with the local MMseqs2 build and checks that it reproduces `data/splits.json` (40%) or `data/splits_homology70.json` (70%); `--keep-tsv` saves the cluster assignments. |
| `scripts/make_seed_splits.py` | Writes the re-seeded splits `data/splits_seed{1,7,123}.json` (CDS) and `data/splits_tss_disjoint_seed{1,7,123}.json` (TSS) from the tracked clusters; never touches the primary splits. |
| `scripts/build_protein_pairs.py` | Builds the purge's pair table `data/leaks/protein_pairs.tsv` (MMseqs2 all-vs-all on full-length proteins) and its metadata. |
| `scripts/recompute_all.py`, `scripts/recompute_all.sh` | The camera-ready manifest runner: every probe cell and null band, records in `data/v2/`, resumable, `--shard I/N` for parallel shards, `--threads N` for a trial at another thread count. |
| `scripts/diff_records.py` | Key-aware diff of two runs: picks, gate fields, predictions, metrics and the builder-level picks; exit 0 (same), 1 (differs), 2 (unusable input). |
| `scripts/train_logistic_probe.py` | Trains multinomial family5 probes and classification baselines. |
| `scripts/train_probe.py` | Trains Ridge probes from DNA features into GenePT text embeddings. |
| `scripts/train_baseline.py` | Runs 4-mer Ridge baseline cells. |
| `scripts/train_anti_baseline.py` | Runs shuffled-GenePT anti-baseline cells for leakage checks. |
| `src/cluster/mmseqs_cluster.py` | Translates each CDS, runs `mmseqs easy-cluster` and maps every gene to a cluster (a gene with no CDS becomes a singleton). |
| `src/splits/make_splits.py` | The random stratified splitter and the greedy whole-cluster, family-balanced assigner. |
| `src/splits/leaks.py` | The evaluation purge: the rules for each split file (`rules_for`) and each cell's masked genes (`purge_for`). |
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

- `data/splits.json` - the CDS primary split: whole 40%-identity protein clusters, family-balanced 70/15/15.
- `data/splits_homology70.json`, `data/splits_random.json` - the 70% homology split and the random split.
- `data/clusters/homology_id40.tsv`, `data/clusters/homology_id70.tsv` - canonical MMseqs2 cluster assignments (representative, member) behind the two homology splits.
- `data/splits_seed{1,7,123}.json`, `data/splits_tss_disjoint_seed{1,7,123}.json` - re-seeded splits for the seed-sensitivity table.
- `data/leaks/protein_pairs.tsv`, `data/leaks/protein_pairs.json` - the purge's homologous pairs and their build metadata.
- `data/v2/metrics_<split stem>.json`, `data/v2/null_<split stem>.json` - the camera-ready records and null-band runs (written by `recompute_all.sh`; the canonical run is committed); predictions in `outputs/predictions/v2/<split stem>/` (not tracked).
- `data/metrics.json` - appended probe and baseline metrics (per-cell CLI runs).
- `outputs/predictions/<metrics stem>/*.npz` - each cell's stored test predictions (content-addressed names), which the bootstraps rescore; `--pred-dir` overrides the location (gitignored).

## Results

This page states no results. The current numbers live in the records
`data/v2/metrics_<split stem>.json` and in `data/v2/statistics.json`. The paper's
tables come from `scripts/build_paper_tables.py`, and its prose numbers
(`numbers.tex`) from `scripts/build_numbers.py`. To regenerate them, run
`scripts/recompute_all.sh all` (see its header), then the builders in
`docs/stage7-paper-figures-tables.md`.
