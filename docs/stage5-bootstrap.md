# Stage 5: Test-Set Uncertainty and Confirmatory Tests

`scripts/build_statistics.py` produces every interval, paired test and null
band the paper reports. It fits nothing: it rescores the test predictions that
the Stage 3 and 4 cells stored, with the functions in
`src/linear_trainer/stats.py`, and writes `data/v2/statistics.json`.

## Pipeline placement

Run it after `scripts/recompute_all.sh all` has run every probe cell and null
band from a committed tree. It reads the records in `data/v2/` through
`linear_trainer.records` and refuses to start unless:

- `data/v2/run_complete.json` says one whole-manifest run wrote them, at one
  commit and protocol;
- every records file is from the same commit and protocol;
- each split file on disk matches the hash its records carry.

The stored predictions live in `outputs/predictions/v2/` (not tracked); each
record names its file and hash.

Sample input and output shapes are tracked in
`samples/stage5_bootstrap_input.json` and
`samples/stage5_bootstrap_output.json`.

## Relevant Files

| File | What it does |
| --- | --- |
| `scripts/build_statistics.py` | Picks the headline cells on validation, runs the intervals, the confirmatory and exploratory paired tests and the null bands, writes `data/v2/statistics.json`. |
| `src/linear_trainer/stats.py` | Cluster bootstrap, paired bootstrap, Holm adjustment and null bands over stored predictions. |
| `src/linear_trainer/records.py` | Loads and checks the `data/v2/` records; the validation-only picks (`best_pool`, `best_encoder`, `best_nt_kmer`, `best_aa`). |
| `data/v2/statistics.json` | The output (generated, not tracked). |
| `outputs/predictions/v2/<split stem>/` | Stored test predictions, one content-addressed `.npz` per record (not tracked). |

## What it does

For every cell it reports:

1. **Load** the record's stored test predictions and verify them against the
   record's sha256. A refit could land on a different model (another machine,
   thread count or `max_iter`), so nothing is refitted.
2. **Drop** the genes the evaluation purge masks from scoring, and check that
   the rest reproduce the record's test value exactly. The script stops
   otherwise.
3. **Resample** whole groups of test genes 1,000 times with replacement, and
   take the 2.5th and 97.5th percentiles as the 95% interval. Genes that
   share sequence are not independent, so the groups are 40% protein clusters
   for CDS cells on the homology-type splits, and window-and-protein groups
   for every TSS cell and every cell on a disjoint split. Resampling is not
   stratified by family, so the intervals are conservative.

**Paired tests** resample both cells of a comparison with the same groups on
the same scored genes. They report the difference A - B, its 95% interval
and a one-sided p-value for A > B.

**Confirmatory family:** four family5 macro-F1 tests, Holm-adjusted together.
Each side other than ESM-2 650M (a fixed model) is the validation-selected cell.

| Test | Comparison | Split |
| --- | --- | --- |
| T1 | best DNA encoder vs nucleotide k-mer (k chosen on validation) | CDS primary |
| T2 | best DNA encoder vs amino-acid k-mer (k chosen on validation) | CDS primary |
| T3 | ESM-2 650M vs best DNA encoder | CDS primary |
| T4 | best encoder on CDS vs the same encoder on TSS | TSS primary (disjoint) |

A confirmatory cell whose pick sits at a search limit, whose fit or train+val
refit did not converge, or that predicts a single class stops the build.
Every other paired test is exploratory and unadjusted.

**Null bands** replace a single shuffled-label run as "chance": the 2.5-97.5%
range over 200 label-shuffled runs of one cell, each with its full selection
loop. They cover the 4-mer on each primary split and task (the TSS 4-mer on the
TSS primary), plus ESM-2 650M on family5 on the CDS primary.

## What it is not

The intervals capture test-composition sampling uncertainty only. They do not
reflect:

- hyperparameter sensitivity (the selection is not re-run);
- train/validation split variability (the seed splits are reported separately);
- encoder-side variability (the embeddings are taken as given).

## Running it

```bash
uv run python scripts/build_statistics.py
```

Optional flags:

- `--n-iters N`: number of bootstrap iterations (default `1000`). The paper
  table builder refuses any other value.
- `--out PATH`: override the output path (default `data/v2/statistics.json`).

## Output schema

```json
{
  "stamp": {"git_sha": "...", "protocol_hash": "..."},
  "inputs": {"metrics_splits.json": "<sha256>", "...": "..."},
  "n_iters": 1000,
  "seed": 42,
  "confirmatory": {
    "T1 encoder > nucleotide k-mer": {
      "a": "<cell key>", "b": "<cell key>",
      "delta_point": 0.0, "delta_ci95": [0.0, 0.0],
      "p_one_sided": 0.0, "p_holm": 0.0,
      "n_test": 0, "n_groups": 0, "edges": [false, false]
    }
  },
  "intervals": {
    "splits.json/family5/<source>": {
      "point": 0.0, "ci95": [0.0, 0.0], "kappa_ci95": [0.0, 0.0],
      "n_test": 0, "n_groups": 0, "n_short_class": 0
    }
  },
  "exploratory": {"<split> <task>: <A> > <B>": {"delta_point": 0.0, "...": "..."}},
  "null_bands": {
    "splits.json/family5/kmer": {
      "median": 0.0, "band95": [0.0, 0.0], "n": 200,
      "n_refit_nonconverged": 0, "n_edge": {"plateau": 0, "limit": 0, "nonconverged": 0}
    }
  }
}
```

The paper tables (`scripts/build_paper_tables.py`) and figures read this file,
and refuse it if any records file has changed since it was written.
