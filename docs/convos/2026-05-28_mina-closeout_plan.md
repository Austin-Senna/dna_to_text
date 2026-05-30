# Plan — close out the three CPU-only journal-hardening items (MINA #1/#3/#10)

_Session: 2026-05-28 · branch: `revision/journal-hardening` · plan-mode artifact_

> This is the approved plan that drove the 2026-05-28 closeout session. Session
> summary alongside this file as `2026-05-28_mina-closeout_summary.md`.

## Context
The `revision/journal-hardening` branch is **substantively complete**: homology split (#1 primary 40%),
composition + ESM-2 baselines (#2), Ridge α→R² (#3), TSS coding-overlap quantification + collapse gate
(#4), paired-bootstrap CIs and seed sensitivity (#10). Three small **CPU-only** items remain that close
out the reviewer's technical asks with no Austin dependency. Doing them makes the branch fully
reviewer-complete on the analysis side, leaving only the Austin-gated manuscript reframe (#7) and the
human handoff (send Austin `docs/notes/tss_homology_collapse_brief.md`, #6).

All three are cheap because the heavy artifacts already exist on disk: per-α sweep data in
`data/metrics_homology.json`, per-sample feature parquets for every CDS/TSS arm, and the unprobed
`data/splits_homology70.json`. No re-embedding, no GPU.

---

## Item 1 — α-selection sensitivity table (MINA #3)
**Goal:** show the GenePT-regression conclusions (encoder ranking by test R²) do **not** flip whether
Ridge α is selected by validation mean-cosine vs validation macro-R².

**Key finding from inspection:** in `data/metrics_homology.json`, each regression cell stores
`alpha_sweep = [{alpha, mean_cosine, r2}, …]` (both are *validation* metrics) plus a single top-level
`test_r2_macro` for the **R²-selected** α. Across the 53 regression cells, **50 have identical α under
cosine- vs R²-selection** → their test R² is trivially unchanged. **Only 3 cells differ** (all encoder
`linear_probe` cells), and for those the test R² at the cosine-selected α is *not* stored, so it needs a
tiny re-fit.

**Approach:**
1. New analysis script `scripts/alpha_sensitivity.py`:
   - Load `data/metrics_homology.json`; for each regression cell compute `α_cos = argmax val mean_cosine`
     and `α_r2 = argmax val r2`. Identify the cell's encoder/baseline label (use `feature_source`, and for
     generic `linear_probe` cells map via `run_id`/`dataset` stem — `_harvest()` in `seed_sensitivity.py:72`
     shows the `Path(e["dataset"]).stem` trick).
   - For the ≤3 cells where `α_cos ≠ α_r2`, re-fit Ridge at `α_cos` on the cached embeddings and record
     test R². Reuse the existing fit path — `src/linear_trainer/probe.py` `sweep_alpha` (selection logic at
     `probe.py:79-80`: `key = "r2" if select_by=="r2" else "mean_cosine"; best = max(results, key=…)`).
     The per-α models can be re-fit directly from the cell's parquet; no new training protocol.
2. Emit `analysis/alpha_sensitivity/alpha_selection_table.{csv,md}`: one row per encoder/baseline with
   `α_cos, test_R²(cos)`, `α_r2, test_R²(r2)`, and the two implied rankings side by side, demonstrating the
   ordering is stable.

**Files:** new `scripts/alpha_sensitivity.py`; reuse `src/linear_trainer/probe.py`. **No tracked-file
clobber** (read-only over metrics + ephemeral re-fits). Output under `analysis/alpha_sensitivity/`.

---

## Item 2 — CDS-vs-TSS within-encoder paired bootstrap (MINA #10)
**Goal:** report paired-difference 95% CIs for **CDS arm vs TSS arm within each DNA-LM encoder**,
confirming the TSS arm is significantly worse (the reviewer asked specifically for this pair).

**Approach:** edit `scripts/bootstrap_test_uncertainty.py`:
- TSS sources auto-register in `DATASET_PATHS` when their parquet exists (`tss_<enc>_<variant>`); the paired
  bootstrap fits both probes per replicate from the parquets and aligns by `ensembl_id` via
  `_common_alignment` — so **no per-sample predictions need pre-computing**.
- Append CDS-vs-TSS pairs to `PAIRED_CLS` (lines ~244-252) and `PAIRED_REG` (lines ~255-264), tuple format
  `(label, dataset_A, hparam_A, dataset_B, hparam_B)` where hparam is C (cls) / α (reg). For each of the 4
  DNA-LMs (`dnabert2`, `nt_v2`, `gena_lm`, `hyena_dna`): pair the encoder's best CDS pooling vs its best TSS
  pooling, e.g. `("dnabert2 CDS - TSS", "dnabert2_meanD", C, "tss_dnabert2_meanmean", C)`.
- Look up each source's best C/α from `data/metrics_homology.json` (don't hardcode blindly — match the
  pooling actually used in the headline tables). **Guard:** skip any encoder whose TSS parquet is absent.
- Run `uv run scripts/bootstrap_test_uncertainty.py --paired` (seed 42, n_iters 1000) → regenerates
  `data/bootstrap_metrics.json`.

**Files:** `scripts/bootstrap_test_uncertainty.py`. **Clobber note:** the run rewrites the committed
`data/bootstrap_metrics.json`; since it's deterministic (fixed seed), the existing ESM-2/composition pairs
must reproduce **identically** — verify the diff is purely additive (new CDS-vs-TSS keys only) before
committing.

---

## Item 3 — 70%-identity supplementary split (MINA #1)
**Goal:** re-probe the **headline cells** on the stricter 70%-identity homology split
(`data/splits_homology70.json` exists, same `{train,val,test,classes}` schema, never probed) to show the
primary-split conclusions hold under a stricter homology threshold.

**Approach (headline cells only, per user choice):** reuse `seed_sensitivity.py`'s proven machinery rather
than the full `rerun_on_split.py` matrix.
- New driver `scripts/probe_homology70.py` that **imports** `_run_cells` and `_harvest` from
  `seed_sensitivity` (single source of truth for the headline cell list:
  `CLS_CELLS = [nt_v2_meanG, aa2, kmer, tss_dnabert2_meanmean, esm2_150m, esm2_650m]`,
  `REG = [dnabert2_meanD, tss_dnabert2_meanmean, esm2_150m, esm2_650m] + aa3`).
- Mirror the backup→swap→restore pattern (`seed_sensitivity.py:104-121`): `backup = SPLITS.read_bytes()`;
  overwrite `data/splits.json` with `splits_homology70.json` bytes; in `try`, call
  `_run_cells(DATA/"metrics_homology70.json")`; in `finally`, restore `backup`. **Output goes to
  `metrics_homology70.json`, NOT the canonical `metrics_homology.json`.**
- Print a 70%-vs-40% headline comparison (cls macro-F1 / reg R² per cell) via `_harvest` on both files;
  write it to `docs/notes/homology70_supplementary.md`.

**Gotchas / clobber handling (critical):**
- Classification cells write confusion matrices to a **fixed** path `data/confusion_5way_{source}.json`
  (`train_logistic_probe.py:~126`), clobbering the tracked 40%-split matrices. After the run:
  `git checkout -- data/confusion_5way_*.json` to restore tracked ones, and `git status --porcelain`
  + `rm` any **newly-created untracked** confusion matrices.
- `splits.json` is git-tracked too, so even if the `finally` restore is skipped it's recoverable via
  `git checkout -- data/splits.json` — but the `finally` should handle it.

**Files:** new `scripts/probe_homology70.py` (imports from `seed_sensitivity.py`). New outputs:
`data/metrics_homology70.json`, `docs/notes/homology70_supplementary.md`.

---

## Sequencing
Do in ascending risk order: **Item 1** (read-only-ish, no clobber) → **Item 2** (single-file regen, verify
additive) → **Item 3** (split swap + confusion-matrix clobber/restore). Commit each item atomically.
After all three, update `docs/notes/tss_homology_collapse_brief.md` and
`docs/notes/RESUME_journal_hardening.md` to mark #1/#3/#10 fully done, and refresh memory
`project_journal_hardening.md`.

## Verification
- **Item 1:** open `analysis/alpha_sensitivity/alpha_selection_table.md`; confirm the encoder ranking by
  test R² is the same under both α-selection rules (it should be — only 3 cells differ at all, and those
  are near-zero-R² baselines unlikely to reorder the leaders). Sanity-check the 3 re-fit test-R² values are
  finite and close to the stored R²-selected values.
- **Item 2:** `uv run scripts/bootstrap_test_uncertainty.py --paired`; inspect `data/bootstrap_metrics.json`
  `paired` block — new CDS-vs-TSS entries present with `frac_A_gt_B ≈ 1.0` and CIs excluding 0 for CDS>TSS;
  `git diff data/bootstrap_metrics.json` shows only additive keys (existing pairs byte-identical).
- **Item 3:** `uv run scripts/probe_homology70.py`; confirm `metrics_homology70.json` written,
  `metrics_homology.json` untouched (`git diff` clean for it), and `git status` clean for
  `data/confusion_5way_*.json` after restore. Eyeball the 70%-vs-40% table: ESM-2 still top, AA-composition
  ties/beats best DNA-LM, TSS arm at/near floor.

## Out of scope (do NOT start — Austin-gated)
- #6 send Austin the brief (human action for Hayden).
- #7 manuscript reframe in the `dna_to_text_paper` submodule — explicitly held for Austin's steer.
- #4b masked-TSS control — SKIPPED by recorded unanimous consensus; do not revive unless a reviewer insists.
- Writeup items #4/#5 (pooling-rule statement, de-emphasize best-cell) are manuscript text, folded into #7.
