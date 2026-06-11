# Plan B — ICSB reframe, data-sync pass (Abstract / Results / Figures)

_Branch: `icsb` (in `dna_to_text_paper` submodule). Date: 2026-05-30._

## Scope
Sync every **number, winner, and figure** in the front-of-paper to the committed homology
tables. Owns: `abstract.tex`, `results.tex`, the five figures + their captions, and the
figure-generating scripts. Coordinates with Plan A on the Enformer decision and title.

Source of truth: `paper/tables/*.tex` and `data/metrics_homology.json` (170 cells, list of
records keyed by `model`/`encoder`/`feature_source`/`task`; **no Enformer cells**, 20 TSS
cells). Figures regenerate from `data/metrics_homology.json` + `data/per_dim_r2.json` via
`scripts/build_paper_tables.py` / the analysis-artifact builder.

## Review gate (run after EVERY commit)
```
/code-review        # or Agent → codex:rescue with the commit diff
```
Each step: edit → `tectonic main.tex` (RC=0, 0 `[?]`) → eyeball the rendered page →
commit → Codex review → fixup until clean → next. **Sequential; no skipping ahead.**

## Commits

### B1 — Abstract  *(fixes #1)*
- Replace the random-split Results sentence (NT-v2 meanD 0.828/0.821 vs 0.672/0.702).
- New headline from `family5_main.tex`/`ridge_main.tex`/`protein_comparison.tex`:
  homology split; best DNA encoder NT-v2 meanG 0.727; AA-2mer composition ties at 0.735;
  ESM-2 650M 0.960 / R² 0.181; TSS collapse. Add "homology-aware split" to Motivation.
- **Commit msg:** `abstract: rewrite results to homology-split headline numbers`

### B2 — Results §classification  *(fixes #2)*
- `results.tex:4`: fix all numbers (0.828→0.727 meanG, +0.119→+0.120, 4-mer 0.672→0.633,
  shuffled 0.208/0.048→0.224/0.035).
- Add the composition baselines (codon, AA 1/2/3-mer) — and state that **AA-2mer is the
  best non-control cell** (it's bolded in `family5_main.tex`); stop calling NT-v2 "the
  strongest." Fix the per-class immune/ion deltas against the homology confusion data.
- **Commit msg:** `results: sync classification prose + composition baselines (homology)`

### B3 — Results §regression + per-dim  *(fixes #3)*
- `results.tex:20`: AA-3mer R² 0.090 is the bolded best; DNABERT-2 **meanD** 0.077 (not
  meanG 0.210); 4-mer 0.045; NT-v2 0.052. Fix winner + Δ + cosine band.
- `results.tex:29` per-dim paragraph: recompute "X/1536 positive, median, top-N dims,
  macro mean" from the homology `per_dim_r2.json` (current 1,527 / 0.190 / 429 / 0.210 are
  random-split).
- **Commit msg:** `results: sync regression prose + per-dim diagnostic (homology)`

### B4 — Results §pooling + §context  *(fixes #7, #8)*
- §pooling (`results.tex:38`): NT-v2 span, HyenaDNA clsmean/meanG numbers from
  `s_pooling_full.tex`.
- §context (`results.tex:47`): TSS 4-mer κ 0.033 (not 0.247); CDS→TSS drop magnitude per
  `cds_tss.tex`; align metric (κ) with table; keep paired-bootstrap + seed refs (already
  correct).
- **Commit msg:** `results: sync pooling + substrate-context prose (homology)`

### B5 — Regenerate figures + captions  *(fixes #6)*
- Rebuild from homology data: `confusion_best_family5`, `umap_best_family5`,
  `pooling_heatmap_family5`, `model_tradeoff_f1_vs_r2`, `per_dim_r2_distribution`.
- Fix captions: confusion/UMAP say "NT-v2 **meanD**" → headline cell is now meanG; remove
  embedded random numbers (0.210, 0.828). (Per repo convention: no embedded figure titles —
  caption is the title; drop any matplotlib set_title.)
- **Commit msg:** `figures: regenerate from homology split; fix captions`

### B6 — Enformer in results + title  *(fixes #5, #10)*
- Apply the Plan-A Enformer decision to `results.tex:47` (drop the trunk κ/R²/0.545 line or
  caveat it). Keep consistent with `cds_tss.tex` (no Enformer row).
- Title (`main.tex`): decide with lead author whether "reveals coding-sequence family
  signal" should pivot toward the benchmark / protein-substrate (ESM-2) framing now that
  "beyond k-mers" is gone. Separate sign-off; commit only if approved.
- **Commit msg:** `results+title: finalize enformer handling and title framing`

## Done when
Abstract + all Results subsections + all five figures match the committed tables exactly;
`tectonic` RC=0, 0 unresolved citations; every Codex review clean; Plan A merged/rebased so
Discussion and Results tell one consistent homology story.

## Cross-plan coordination
- **Enformer** (A2 ↔ B6) and **title** (B6) are shared decisions — settle once, apply in both.
- Run **A first or in parallel**, but do a final joint `/code-review` of the whole diff after
  both land, to catch any remaining text/table drift.
