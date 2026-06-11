# Plan A — ICSB reframe, text-truth pass (Discussion / Conclusion / Methods / Intro)

_Branch: `icsb` (in `dna_to_text_paper` submodule). Date: 2026-05-30._

## Scope
Fix the claims that are **false or dangling** regardless of figure/table regeneration.
These edits depend only on the already-committed homology tables (`paper/tables/*.tex`)
and `data/metrics_homology.json` — no figure rebuild, no number recompute beyond reading
the committed tables. Owns: `discussion.tex`, `methods.tex`, `introduction.tex`,
`bibliography.bib`. Does **not** touch `abstract.tex`, `results.tex`, or figures (Plan B).

Source of truth for every number: `paper/tables/{family5_main,ridge_main,cds_tss,
protein_comparison,s_*}.tex`.

## Review gate (run after EVERY commit)
```
# from repo root, codex plugin available in this env
/code-review               # or: Agent → codex:rescue with the commit diff
```
Each step: make edit → `tectonic main.tex` (RC=0, 0 `[?]` in PDF) → commit → Codex review →
address findings in a fixup commit → re-review until clean → next step.
**Do not start step N+1 until step N's review is clean.**

## Commits

### A1 — Kill "beyond composition"; state composition parity  *(fixes review #4)*
- `discussion.tex` "Supported claim": NT-v2 +Δκ over 4-mer is real, but AA-2mer
  (0.735/0.710) **ties** NT-v2 (0.727/0.704) on classification and AA-3mer (R² 0.090)
  **beats** the best DNA encoder (0.077) on regression. Reframe to "matches but does not
  exceed translated-composition baselines."
- `conclusion` (`discussion.tex:15`): delete "beyond simple composition"; restate as
  homology-controlled parity with composition.
- Verify wording against `family5_main.tex` + `ridge_main.tex` before committing.
- **Commit msg:** `discussion: reframe supported claim to composition parity (homology split)`

### A2 — Resolve Enformer  *(fixes #5, #12)*
- Decision required (ask user / lead): Enformer has **0 cells** in `metrics_homology.json`
  and **no row** in `cds_tss.tex`. Either (a) drop all Enformer mentions, or (b) keep with
  an explicit "not re-run on the homology split" caveat (mirroring the TSS-4-mer-regression
  caveat already in `cds_tss.tex`).
- Apply chosen option in `discussion.tex:5` (the "Enformer trunk 0.439/0.545" sentence),
  `methods.tex:43-44` (trim the half-paragraph if dropping), `introduction.tex:9`
  (Enformer listed as headline comparator).
- **Commit msg:** `enformer: <drop|caveat> mentions absent from homology results`

### A3 — TSS magnitude + metric consistency  *(fixes #8, #13)*
- `discussion.tex:5`: "drop by ~half" → the homology drop is to ~a quarter; TSS 4-mer is
  **κ 0.033** (not 0.247). Pull exact ranges from `cds_tss.tex` / `s_cds_tss_paired.tex`.
- Pick ONE metric (κ vs macro-F1) for the substrate narrative and use it consistently with
  the table (table reports κ).
- Optionally strengthen: homology TSS sits near the 0.224 chance floor — the collapse story
  is *stronger* now; say so.
- **Commit msg:** `discussion: correct TSS-collapse magnitudes to homology numbers`

### A4 — Limitations cleanup  *(fixes #11)*
- `discussion.tex:21`: remove/soften "CIs do not cover split-seed variance" — a 4-seed
  sensitivity table now exists (`s_seed_sensitivity.tex`, Supp. Table ref
  `tab:s-seed-sensitivity`). Cite it instead.
- Sweep the limitations para for any other now-addressed item (paralog splits already
  removed in an earlier commit).
- **Commit msg:** `discussion: update limitations (seed sensitivity now reported)`

## Done when
All four commits reviewed-clean by Codex, `tectonic` RC=0 with 0 unresolved citations, and
no Discussion/Methods/Intro sentence references a number absent from the committed tables.
Hand off to Plan B owner; note any Enformer decision so Plan B's results prose matches.
