# Design: Homology-data refresh of paper tables (Python-generated)

_Date: 2026-05-30 · Parent branch: `revision/journal-hardening` · Submodule: `dna_to_text_paper` on `main` (47bc26e)._

## Goal

Regenerate every table in the manuscript (`dna_to_text_paper`) from the new
**homology-aware split** data produced on `revision/journal-hardening`, using a
**Python generator** so numbers are imported from the data rather than
hand-transcribed. Add the new baselines (ESM-2 150M/650M, AA-composition
aa1/2/3, codon, kmer6, GC) and three new supplementary robustness tables.

## Scope (decided in interview)

- **Tables + appendix data only.** The Results *narrative* prose is left as-is for
  now, even where it will numerically contradict the regenerated tables (e.g.
  prose still calls NT-v2 0.828 the headline). A prose reframe is a separate
  follow-up task — **out of scope here.**
- **Homology split is primary; the random split is dropped entirely** from the
  tables (not moved to appendix).
- **α is selected by validation macro-R²** (already how `metrics_homology.json`
  was generated: every regression record has `select_by: "r2"`). No
  cosine-vs-R² sensitivity table; R²-selection is simply the canonical pipeline.

### Two sanctioned prose touches (everything else in prose untouched)
1. One sentence in `results.tex` noting the seed-sensitivity robustness check and
   pointing to its appendix table.
2. Minimal **factual-consistency fixes** in the appendix "Deterministic settings"
   so it does not misdescribe its own (now homology/R²) tables:
   - split description: random seed-42 stratified (2270/487/487 from
     `dataset.parquet`) → homology MMseqs2 cluster split, min-seq-id 0.4,
     coverage 0.8, 1751 clusters, **2271/486/487** train/val/test.
   - "Ridge-to-GenePT probes ... selecting by validation mean cosine" →
     "selecting by validation macro-R²".
   - appendix intro "frozen 70/15/15 split" wording: clarify it is the
     homology-clustered 70/15/15 split.

## A. Generator script

**`scripts/build_paper_tables.py`** (parent repo).

- **Reads** (tracked, parent repo):
  - `data/metrics_homology.json` — primary 40% split, 170 records (54 family5
    cells, 53 regression cells, plus binary tasks we ignore here).
  - `data/metrics_homology70.json` — 70%-id supplementary split.
  - `data/seed_sensitivity/summary.json` — seeds {1,7,42,123} → {cls,reg} →
    {source: value}.
  - `data/bootstrap_metrics.json` — has top-level `paired` block for CDS-vs-TSS
    difference CIs.
- **Writes** LaTeX fragments into the submodule:
  `dna_to_text_paper/paper/tables/*.tex`. Each fragment is the **data-row body
  only** (data rows + any `\multicolumn` section separators). The paper's own
  `.tex` files keep the `\begin{table}` / `\processtable{caption}` /
  `\begin{tabular*}` / column header / `\toprule` / source-note / `\botrule`
  wrappers and add one `\input{paper/tables/<name>.tex}` where the rows go.
- Generated fragments are **committed into the submodule** so Overleaf builds
  without the parent repo present.

### Source normalization (the core of the script)

The data encodes the "source" of a cell three different ways:
- classification (`task` present): `feature_source`, e.g. `nt_v2_meanD`,
  `aa2`, `esm2_650m`, `tss_dnabert2_meanmean`, `enformer_tss_4mer`.
- regression linear-probes (`model == "linear_probe"`, no `task`): identified by
  `dataset` filename, e.g. `dataset_dnabert2_meanG.parquet`,
  `dataset_esm2_650m.parquet`.
- regression baselines (`model` like `kmer_baseline_4`, `aa_baseline_3`,
  `codon_baseline`, `gc_baseline`).

A single `normalize_source(record) -> {raw_id, encoder_display, pooling_display,
category, sort_key}` resolves all three. Categories and display names:

| raw id pattern | encoder display | category |
|---|---|---|
| `kmer` (4-mer) | CDS 4-mer | dna-composition |
| `kmer6` | CDS 6-mer | dna-composition |
| `codon` | Codon freq. | dna-composition |
| `gc` | GC + length | dna-composition |
| `aa1`/`aa2`/`aa3` | AA 1/2/3-mer | aa-composition |
| `dnabert2_*`,`nt_v2_*`,`gena_lm_*`,`hyena_dna_*` | DNABERT-2 / NT-v2 / GENA-LM / HyenaDNA | dna-lm-cds |
| `tss_<enc>_*` | same names | dna-lm-tss |
| `esm2_150m`/`esm2_650m` | ESM-2 150M / 650M | protein-lm |
| `enformer_*` | Enformer ... | enformer |
| shuffled | Shuffled labels / Shuffled-Y | control |

- Pooling parsed from the suffix (`meanmean`, `specialmean`, `maxmean`,
  `clsmean`, `meanD`, `meanG`); composition baselines and ESM-2 have no pooling
  (`---`).
- Number formatting matches existing tables: main tables 3 decimals, appendix
  4 decimals; deltas carry an explicit `+`/`-` sign; α printed as in current
  tables.
- **Bolding**: script emits `\textbf{}` on the best non-control value per metric
  column (matches "Best non-control value is bolded" convention). With homology
  data this moves the bold to ESM-2 650M.

### "Best cell per source" selection
For the main best-cell tables, per source pick the pooling with the highest
metric (macro-F1 for classification, R² for regression) — same rule as the
existing tables, applied over the homology cells. Encode the rule in code (no
ad-hoc picks).

## B. Main-text tables (`results.tex`)

All numbers from `metrics_homology.json`.

- **Table 1 `tab:family5-main`** — rows (in order): Shuffled labels · CDS 4-mer ·
  CDS 6-mer · Codon · GC+length · AA 1-mer · AA 2-mer · AA 3-mer · DNABERT-2 ·
  NT-v2 · GENA-LM · HyenaDNA · ESM-2 150M · ESM-2 650M. (DNA-LMs/ESM use best
  pool.) Bold = ESM-2 650M.
- **Table 2 `tab:ridge-main`** — same source set, R²-selected α, ESM-2 bolded.
- **Table 3 `tab:cds-tss-comparison`** (substrate ablation) — kept as the 4
  DNA-LMs + Enformer, homology numbers, CDS vs TSS sections. **ESM-2 is not
  added** (protein/CDS-only, no TSS analog). Composition baselines stay as the
  within-context 4-mer reference rows already present.

Known accepted side effect: Tables 1–3 will numerically contradict the
surrounding Results prose until the separate reframe task runs.

## C. Appendix tables (`appendix.tex`)

- **A1 `tab:s-pooling-full`** (classification full matrix) — homology numbers;
  CDS section gains rows for aa1/aa2/aa3, codon, kmer6, gc, ESM-2 150M/650M; TSS
  section refreshed to homology. Section group separators kept.
- **A2 `tab:s-regression-full`** (regression full matrix) — same additions,
  R²-selected α, homology numbers. Keep the `Mean cosine` column (still a
  reported sanity metric) and the `α` column.
- **A3 (new) `tab:s-seed-sensitivity`** — headline cells (ESM-2 650M, ESM-2
  150M, best DNA-LM, best AA-composition, a TSS cell) × seeds {42,1,7,123},
  showing macro-F1 and R² ranges. Source: `seed_sensitivity/summary.json`.
- **A4 (new) `tab:s-homology70`** — headline cells re-probed on the 70%-identity
  split, demonstrating the leakage gradient (every cell higher than 40%; TSS
  rises ~0.326→0.505). Source: `metrics_homology70.json`.
- **A5 (new) `tab:s-cds-tss-paired`** — per-encoder paired-bootstrap CDS−TSS
  difference: ΔF1 [95% CI], ΔR² [95% CI], P(CDS>TSS). Source:
  `bootstrap_metrics.json` `paired` block.

## D. Files touched

- **New:** `scripts/build_paper_tables.py` (parent).
- **New:** `dna_to_text_paper/paper/tables/{family5_main,ridge_main,cds_tss,
  s_pooling_full,s_regression_full,s_seed_sensitivity,s_homology70,
  s_cds_tss_paired}.tex` (generated row-body fragments).
- **Edited (submodule):** `paper/results.tex`, `paper/appendix.tex` — replace
  hand-typed rows with `\input`; add 3 new appendix table wrappers; one
  seed-sensitivity sentence; protocol-fact fixes. Possibly `paper/methods.tex`
  if its dataset-composition/split table reflects split sizes (verify; update
  split counts only).

## E. Verification

1. `uv run scripts/build_paper_tables.py` regenerates all fragments
   idempotently.
2. Rebuild the PDF via the submodule `makefile`; confirm clean compile (no
   missing `\input`, no overfull errors introduced beyond existing).
3. Spot-check ≥4 generated numbers against the raw JSON (e.g. ESM-2 650M F1
   0.960, AA-2mer F1 0.735, DNABERT-2 meanG R² from homology, TSS DNABERT-2 at
   floor).
4. Confirm bolding lands on ESM-2 650M in Tables 1–2.

## Non-goals

- No Results/Discussion narrative reframe.
- No change to figures (confusion matrix, UMAP, per-dim R², heatmap, tradeoff)
  unless a figure caption cites a number that moved — out of scope; note if
  found.
- Binary tasks (`tf_vs_gpcr`, `tf_vs_kinase`) in the data are not surfaced.
- No commit/push unless the user asks.
