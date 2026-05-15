# Conference resubmission plan & status

Branch: `revision/tss-and-gene-scope`
Started: 2026-05-15
Origin: post-submission reviewer feedback on the COMS W4761 submitted version (`origin/main` HEAD `7e32954`, submodule pointer `73a53bf`). Resubmission target is a real conference, not the class.

This file tracks the plan and live status so any future session can pick up without re-deriving context. Update the **Status** column as items move.

---

## Context

Two reviewer feedback items, both flagged "not extremely important", both of which the paper itself already calls out as open questions in §Limitations (`dna_to_text_paper/paper/discussion.tex:19`):

1. **Expand the TSS experiment to all encoders.** Today only NT-v2 is run on TSS windows; DNABERT-2, GENA-LM, HyenaDNA are CDS-only. Paper says: *"whether the substrate effect is encoder-general or NT-v2-specific is open."*
2. **Include more / all genes** if feasible. Today: 3,244 genes across 5 families. Paper says: *"families with weaker domain signatures might yield smaller encoder-vs-k-mer gaps."*

Compute available: constrained Colab (same as before) **plus local RTX 5060 (16 GB VRAM)** — enough for overnight inference-only encoder runs.

---

## Recommendation summary

| Item | Decision | Estimated cost | Status |
|---|---|---|---|
| TSS expansion (all 4 encoders) | **Do it** | ~10–15 GPU-h on RTX 5060 | not started |
| Gene-set expansion | **Investigate first; only run if paired with paralog-aware split** | 2–4 h analysis, then decide | not started |
| Paralog-aware splitting (MMseqs2/CD-HIT) | **Only if we run gene expansion** | ~1 day to wire up | not started |

---

## Phase 1 — TSS expansion

### Gap

Existing: `data/dataset_tss_nt_v2_*.parquet` (5 pooling variants) + `data/dataset_tss_4mer*` + `data/dataset_tss_enformer*`.

Missing: TSS embeddings for **DNABERT-2**, **GENA-LM**, **HyenaDNA**.

### Run order (smallest → largest, fail-fast on smallest)

1. **HyenaDNA** (6.6 M params, 8192-token / 512-stride) — overnight #1.
2. **DNABERT-2** (117 M, 510 / 64) — overnight #2.
3. **GENA-LM** (110 M, 510 / 64) — overnight #3.

### Commands per encoder

```bash
uv run python scripts/run_tss_multi_pool_extract.py --encoder <name> --device auto
uv run python scripts/build_tss_pooling_datasets.py --encoder <name>
uv run python scripts/train_logistic_probe.py --dataset tss_<name>_<best_pool> --task family5
uv run python scripts/train_probe.py --dataset tss_<name>_<best_pool> --task genept
```

After all three encoders complete, extend `scripts/bootstrap_test_uncertainty.py` `HEADLINE_CLS` / `HEADLINE_REG` (lines 59–77) with the three new TSS cells and run:

```bash
uv run python scripts/bootstrap_test_uncertainty.py
```

### Existing utilities to reuse — do NOT rewrite

| Utility | File | Notes |
|---|---|---|
| TSS extraction (multi-pool) | `scripts/run_tss_multi_pool_extract.py:31` | Already supports all 4 encoders via `--encoder` |
| TSS pooling dataset builder | `scripts/build_tss_pooling_datasets.py:24` | Same |
| Encoder model loader | `src/data_loader/model_registry.py:38–83` | All 4 encoder specs already defined |
| Bootstrap | `scripts/bootstrap_test_uncertainty.py` | Just extend the headline tables |

### Paper edits after results land

- `dna_to_text_paper/paper/results.tex:53–58` (§3.3) — replace single-encoder TSS sentence with a multi-encoder comparison; update Figure 4 caption.
- `dna_to_text_paper/paper/methods.tex:38–42` (§TSS context) — drop the "given limited GPU resources, … NT-v2 only" caveat.
- `dna_to_text_paper/paper/discussion.tex:19` — rewrite the "TSS run is a targeted context ablation" paragraph; answer the encoder-general-vs-NT-v2-specific question.
- `dna_to_text_paper/paper/abstract.tex` — update headline TSS numbers only if NT-v2 is no longer best on TSS.
- `docs/stage4-2-tss-encoders.md` — update encoder list.

### Verification

1. **NT-v2 sanity-check**: re-run reproduces 0.447 macro-F1 on TSS. Any drift = pipeline regression.
2. **Anti-baseline check**: shuffled-label TSS cell lands near chance (~0.20 macro-F1).
3. The *result itself* — whether all four encoders cluster near 0.45 (substrate-dominance is encoder-general) or split (NT-v2 was specifically strong) — is the finding, not a bug.

---

## Phase 2 — Gene-set scoping

### Step 2a — Quantify recoverable lift (analysis only, ~2–4 h CPU)

Produce three numbers, write to `docs/notes/gene_scope_analysis.md`:

1. **GenePT pickle membership ceiling.** Load GenePT v2 pickle (Zenodo 10.5281/zenodo.10833191; path in `scripts/prepare_data.py`). Count how many symbols overlap the 5 HGNC family regexes vs. the current 3,244.
2. **First-family-wins paralog drops.** Re-run the family iteration in `src/data_loader/dataset_loader.py:176–183` without `seen_ensembl` exclusion; count reassignments.
3. **CDS-fetch failures.** Grep `logs/` and `data/sequences/` for genes with Ensembl IDs but no CDS file.

### Step 2b — Decide based on lift

| Recoverable lift | Action |
|---|---|
| < 15 % | **Skip.** Document in §Limitations; mark this entry done with the reason. |
| 15–50 % | **Run only if paired with paralog-aware split.** N gain without leakage fix is hollow for a reviewer. |
| > 50 % | **Run with paralog-aware split.** Becomes the headline methodology upgrade. |

### Why we are NOT pursuing "all ~20 k protein-coding genes"

Paper's central claim is framed around *"five broad protein-family categories that all carry strong, well-characterised domain motifs"* (`methods.tex:5`). Expanding to all protein-coding genes:

- Forces a new family ontology or "other" catch-all.
- May *weaken* the encoder-vs-k-mer gap on weak-motif families — explicitly predicted in `discussion.tex:19` as an open question.
- Requires rerunning all 4 encoders × 6 poolings on ~6× the data: 3–5 days of GPU time, plus story-rewrite.

For this revision cycle that's a separate paper. Track in §"Out of scope" below.

### Files that would change if Step 2b says "run"

- `scripts/prepare_data.py`, `src/data_loader/dataset_loader.py` — relax filter order; optionally add paralog-aware split.
- `scripts/make_splits.py` — new clustered-split mode (MMseqs2 or CD-HIT at 50 % identity).
- Every downstream probe / bootstrap output — regenerate.
- `dna_to_text_paper/paper/methods.tex` §Dataset + Table 1; all results numbers; §Discussion paralog-leakage paragraph.

---

## Live status tracker

| ID | Item | Owner | Status | Notes |
|---|---|---|---|---|
| T-1 | Run HyenaDNA TSS extraction | — | not started | overnight on RTX 5060 |
| T-2 | Run DNABERT-2 TSS extraction | — | not started | |
| T-3 | Run GENA-LM TSS extraction | — | not started | |
| T-4 | Build pooling datasets for T-1..T-3 | — | not started | `build_tss_pooling_datasets.py` |
| T-5 | Train probes (family5 + genept) for T-1..T-3 | — | not started | pick best pool per encoder from `data/metrics.json` |
| T-6 | Extend `bootstrap_test_uncertainty.py` with 3 new TSS cells; rerun | — | not started | |
| T-7 | Update `results.tex` / `methods.tex` / `discussion.tex` for multi-encoder TSS | — | not started | |
| G-0 | Extract GenePT_emebdding_v2/ from Austin's data.zip | — | **done 2026-05-15** | unzipped to repo root; gitignored; pickle has 93,800 symbol keys, 1,536-d each |
| G-1 | Quantify GenePT pickle ceiling within 5 family regexes | — | unblocked | pickle ready at `GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle`; write to `docs/notes/gene_scope_analysis.md` |
| G-2 | Count paralog drops under first-family-wins | — | unblocked | re-run iteration in `src/data_loader/dataset_loader.py:176–183` without `seen_ensembl` |
| G-3 | Count CDS-fetch failures | — | not started | grep `logs/` and `data/sequences/`; cheap CPU work |
| G-4 | Decide on gene-expansion based on G-1..G-3 | — | not started | rule in §Step 2b above |
| G-5 | Wire paralog-aware split (only if G-4 = run) | — | not started | MMseqs2 or CD-HIT at 50% |
| P-1 | Rebuild `main.pdf`, bump submodule, commit + push | — | not started | after T-7 lands |

Update the **Status** column to `in progress` / `done <date>` as items move. Add a one-line **Notes** entry on completion so the next session sees what was actually run.

---

## Out of scope for this revision

- **All ~20 k protein-coding genes** — separate paper.
- **Per-dimension R² distribution for GenePT regression** — A·20 in `docs/archive/notes/paper_revision_followups.md`; punt unless an editor asks.
- **Li 2026 SeqCLIP citation date check** — C·21; 5-minute author lookup, can land any time.
- **Additional organisms / per-layer probes / noncoding-regulatory targets** — `discussion.tex` future-work items; not in revision scope.

---

## Pointers for the next session

- Resume from the **Live status tracker** above — find the first `not started` item and run it.
- Two repositories: parent `/home/hayden/dna_to_text` (this branch, `revision/tss-and-gene-scope`) and submodule `dna_to_text_paper` (will work on `paper-draft` branch when paper edits start).
- Detailed plan file (the one used to generate this doc) is at `~/.claude/plans/understand-the-project-existing-starry-yao.md` — same content, repo-external.
- The paper as submitted is at parent `7e32954` → submodule `73a53bf`. The current submodule working tree at `a952cf1` is the post-bootstrap version with reader-feedback edits; this is the version the revision builds on.

### Austin's data snapshot

Hayden's Windows side has Austin's full `data/` snapshot at `/mnt/c/Users/hayde/Downloads/data.zip` (~2.4 GB). Scouted 2026-05-15. Contents:

- `GenePT_emebdding_v2/` — the GenePT v2 pickles + NCBI summary JSONs. **Extracted** to repo root as of 2026-05-15 (gitignored). ada_text pickle has 93,800 keys × 1,536-d.
- `data/sequences/` — same 3,244 ENSG `.fa` files we already have.
- `data/dataset_<enc>_<pool>.parquet` — full CDS pooling sweep for all 4 encoders × 6 pools. Same as ours.
- `data/dataset_tss_nt_v2_*.parquet` — NT-v2 TSS only. **No TSS data for DNABERT-2 / GENA-LM / HyenaDNA**, so Phase 1 still requires running those extractions ourselves.
- `data/chunk_reductions_<encoder>/` — CDS chunk-level intermediates for all 4 encoders. Could rebuild CDS pooling variants without re-running encoders if ever needed. Does **not** include TSS.
- `data/embeddings/` — per-gene `.npy` files (3200 B each = 800 floats). Local copy is empty; not currently needed.

If anything in `data/` gets corrupted or lost locally, restore from this zip rather than re-running the pipeline.
