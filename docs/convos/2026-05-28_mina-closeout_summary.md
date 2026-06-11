# Session summary — 2026-05-28 journal-hardening closeout

_Branch: `revision/journal-hardening` · plan: `./2026-05-28_mina-closeout_plan.md`_

## Goal
Close out the three CPU-only reviewer items (MINA #1/#3/#10) that didn't need Austin or GPU,
leaving the branch fully reviewer-complete on the analysis side. Only Austin-gated work would
remain afterward (send him the decision brief; #10 manuscript text + reframe).

## What happened in order

1. **Recon.** Read `RESUME_journal_hardening.md`, the Austin brief, and `MINA.md` to map the
   pending list. Spawned three parallel Explore agents (one per item) to nail down the exact code
   locations, hyperparameters, and clobber risks. Inspected `metrics_homology.json` to discover that
   only **3 of 53** regression cells select a different Ridge α under cosine vs R² — collapsing #3
   from "build a table from re-runs" to "build it almost entirely from existing data".
2. **Plan.** Confirmed scope (all three items, headline cells only for the 70% split) via
   `AskUserQuestion` and wrote the approved plan to
   `~/.claude/plans/what-s-next-giggly-hickey.md` (copied here as
   `2026-05-28_mina-closeout_plan.md`).
3. **Item 1 → commit `c5bfea2`.** Wrote `scripts/alpha_sensitivity.py`, ran it (re-fit 3 cells with
   `--select-by cosine`, ~5 s total), wrote `analysis/alpha_sensitivity/alpha_selection_table.{md,csv}`.
4. **Item 2 → commit `dfb9e86`.** Made 5 edits to `scripts/bootstrap_test_uncertainty.py`: appended
   5 CDS-vs-TSS pairs to `PAIRED_CLS` and 5 to `PAIRED_REG`, added a `_resolvable()` helper, and
   added a guard clause in each of the two paired loops in `main()`. Ran the full bootstrap
   (~12 min). Verified the diff to `bootstrap_metrics.json` was purely additive — **0 changed
   leaves, 0 removed leaves** — so the existing ESM-2/composition pairs reproduced byte-identically.
5. **Item 3 → commit `13145d3`.** Wrote `scripts/probe_homology70.py` (imports
   `_run_cells`/`_harvest`/`SPLITS` from `seed_sensitivity` so the headline-cell list stays in one
   place). Ran it: swap → run headline cells → restore. Restored the 4 clobbered confusion matrices
   via `git checkout -- data/confusion_5way_*.json`. Wrote
   `docs/notes/homology70_supplementary.md`.
6. **Docs reconciliation → commit `6672603`.** Updated `RESUME_journal_hardening.md` (header,
   pending block, MINA status lines) and added a "Supplementary robustness (added 2026-05-28; closes
   MINA #1/#3/#10)" section to `docs/notes/tss_homology_collapse_brief.md`. Updated the
   `project_journal_hardening.md` auto-memory.
7. **Push.** `git push -u origin revision/journal-hardening`.
8. **(Post-summary) ESM-2 reproducibility patch → commit `d3b8ee3`.** Auditing data/, noticed
   `dataset_esm2_{150m,650m}.parquet` (71 MB total, both under GitHub's per-file limit) were
   local-only — so the ESM-2 #9 headline wasn't reproducible from origin. Added + committed +
   pushed.

## Findings (each item, one line)

- **#3 α-selection sensitivity:** 50/53 regression cells pick the same α under cosine vs R²;
  leader ordering (ESM-2 650M > 150M > AA-3mer > DNA-LMs) **identical** under both rules.
- **#10 CDS-vs-TSS within encoder (paired):** every DNA-LM beats its TSS counterpart;
  P(CDS>TSS) = 1.000 for all pairs; cls ΔF1 +0.14 to +0.45, reg ΔR² +0.02 to +0.07.
- **#1 70%-identity supplementary split:** conclusions stable; every cell *higher* at 70% than
  at 40% (looser threshold → more leakage); TSS climbs 0.326 → 0.505 — a clean **leakage
  gradient** that reinforces the collapse finding.

## Commits this session (5)

```
d3b8ee3  #9 data: commit ESM-2 dataset parquets for reproducibility
6672603  docs: mark MINA #1/#3/#10 closeout items done (alpha-sens, CDS-vs-TSS, 70% split)
13145d3  #1 supplementary 70%-identity homology split: conclusions stable
dfb9e86  #10 CDS-vs-TSS within-encoder paired bootstrap: CDS beats TSS everywhere
c5bfea2  #3 alpha-selection sensitivity: regression ranking stable (cosine vs R^2)
```

All on `origin/revision/journal-hardening`.

## Files created or modified

**New code:**
- `scripts/alpha_sensitivity.py`
- `scripts/probe_homology70.py`

**Edited code:**
- `scripts/bootstrap_test_uncertainty.py` (5 surgical edits)

**New analysis / data outputs:**
- `analysis/alpha_sensitivity/alpha_selection_table.{md,csv}`
- `data/metrics_homology70.json`
- `data/dataset_esm2_{150m,650m}.parquet` (now tracked)
- `data/bootstrap_metrics.json` (regenerated, additive)

**New docs (research deliverables, in `docs/notes/`):**
- `docs/notes/homology70_supplementary.md`

**Edited docs (research deliverables, in `docs/notes/`):**
- `docs/notes/RESUME_journal_hardening.md`
- `docs/notes/tss_homology_collapse_brief.md`

**New session logs (this directory, `docs/convos/`):**
- `docs/convos/2026-05-28_mina-closeout_plan.md`
- `docs/convos/2026-05-28_mina-closeout_summary.md`

## Where the branch stands

**MINA #1/#2/#3/#4/#10 are all technically complete.** What remains is Austin-gated only:

- **#6 (human):** send Austin `docs/notes/tss_homology_collapse_brief.md`.
- **#7 (Austin's call):** manuscript reframe in the `dna_to_text_paper` submodule —
  title/abstract/central claim pivot toward the 3-part finding (CDS = composition;
  TSS = leakage/noise; ESM-2 wins).
- **#10 writeup tail (folds into #7):** pre-specified pooling-selection-rule statement and
  de-emphasize best-cell reporting.

## Process notes

- The GateGuard fact-forcing hook intercepted every Edit/Write this session (each tool call
  blocked once, then succeeded on retry after presenting the 4 required facts). Workable but
  adds friction across ~10 edits. To smooth a future implementation session, the resume doc's
  recommended recovery is to launch with `ECC_GATEGUARD=off` or add
  `pre:edit-write:gateguard-fact-force` (and the bash equivalent) to `ECC_DISABLED_HOOKS`.
- The `context-mode` MCP server disconnected partway through; subsequent ad-hoc data inspection
  used short Python one-liners via `Bash` instead of `ctx_execute`. No impact on the work.
