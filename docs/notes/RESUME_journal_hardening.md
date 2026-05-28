# RESUME — journal-hardening phase (handoff)

Last updated: 2026-05-28. Branch: **`revision/journal-hardening`** — working tree clean, **not pushed** (latest commit `0f4ffe3`). Analysis is substantively COMPLETE; what remains is 3 small CPU items + manuscript text + the Austin-gated reframe (see **Pending**).

## Read these first
- `docs/notes/tss_homology_collapse_brief.md` — **THE decision brief for Austin** (current, complete): 3-part finding + paired CIs + seed sensitivity + decisions. Start here.
- `docs/journal_hardening_plan.md` — the approved plan (5 MINA.md issues, phases, decisions).
- `MINA.md` (repo root, untracked) — the reviewer's 5 source issues (#1,#2,#3,#4,#10).
- `docs/notes/homology_phase2_results.md` — earlier Phase-2 composition brief (superseded by the collapse brief).
- Memory: `~/.claude/projects/-home-hayden-dna-to-text/memory/project_journal_hardening.md`.

## Locked decisions
Homology split = PRIMARY; both GPU optionals (#4b masked-TSS, ESM-2) in scope; MMseqs2 @40%+70%.

## Foundational commits (Phase 1–2)
- `595ac07` Phase-1 code: #3 Ridge α→macro-R² (`probe.py`/`train_probe.py`), `src/protein/translate.py`, composition baselines (`kmer_baseline.featurize_kmer`, `src/composition_baseline/{codon,aa_kmer,gc}`), #10 paired bootstrap.
- `1edd3a9` fair-esm dep + gitignore (tools/, data/annotation/, data/cluster_work/).
- `43cb4c4` #1 homology split: `src/cluster/mmseqs_cluster.py`, `build_cluster_splits` in `src/splits/make_splits.py`, rewritten `scripts/make_splits.py`. Wrote `splits.json` (PRIMARY 40%), `splits_homology70.json`, `splits_random.json` (+`splits_random_original.json` backup).
- `d41fd4f` Phase 2: `scripts/rerun_on_split.py` (125 cells → `data/metrics_homology.json`), bootstrap synthetic sources + homology PAIRED lists, regenerated confusion matrices.

## KEY FINDING (drives the manuscript reframe — Austin's call)
Homology 40% split: NT-v2 beats 4-mer (ΔF1 +0.089 [+0.025,+0.161]) but is **tied with AA-2mer** (ΔF1 −0.013 [−0.070,+0.053]); on regression **AA-3mer beats the best DNA-LM** (ΔR² −0.014 [−0.021,−0.007]). "DNA-LMs beyond composition" does not survive. Cosine ≈0.92 for all cells (R² is the discriminating metric).

## #4a GTF overlap — DONE (committed `6515ecb`)
Built `src/tss_overlap/{gtf,overlap}.py` + `scripts/tss_overlap.py` (numpy boolean-mask, no new dep). Ran over all 3,244 genes (0 skipped); partition buckets sum to 1.0 (asserted). **HEADLINE: the 196,608 bp window is overwhelmingly non-coding context — mean target-CDS = 0.0080 (0.8%), target-intron 0.17, neighbour-intron 0.39, intergenic 0.33.** Reinforces the AA-composition finding (DNA-LMs see ~1% coding). Outputs: `analysis/tss_overlap/{tables,figures}/` (per_gene_overlap.csv, overlap_by_family.{csv,md}, partition_by_family.png, intergenic_intron_distribution.png) — artifacts not committed. GTF cache `data/annotation/gtf_features.parquet` (gitignored). Re-run: `uv run scripts/tss_overlap.py`. (Committed: `src/tss_overlap/` + `scripts/tss_overlap.py`.)

## Stage-0 gate (2026-05-27): TSS arm COLLAPSES on homology split
Ran `rerun_on_split.py --only-tss` (new flag). TSS family5 macro-F1 best **0.326** (chance floor 0.224); TSS GenePT R² best **0.010** (most ≈0/neg). Random split TSS was 0.455/0.122 → that signal was paralog leakage. So the "TSS substrate carries signal" claim does NOT survive homology splitting.

## #4b masked-TSS — SKIPPED (decision recorded)
Three blind independent agents (numbers only) unanimously: don't run masking — homology TSS already at floor (nothing to ablate); masking on the random split is a "trap"; MINA #4's "quantify OR mask" is satisfied by #4a's quantify branch + near-floor homology perf. Minimal hedge if a reviewer insists: one fast encoder, homology split only. Masking infra (`src/tss_overlap/mask.py`, `--mask` on extract) NOT built.

## #9 ESM-2 comparator — DONE (committed `3535159`; paired CIs `03063a9`)
`scripts/run_esm2.py` (CDS→`translate_cds`→ESM-2, mean-over-residues, chunk-and-mean >1022aa, fp16) + `scripts/build_esm2_datasets.py`; registered `esm2_150m`/`esm2_650m` in `train_logistic_probe.DATASET_PATHS`; added `--esm2`/`--only-esm2` to `rerun_on_split.py`. Embeddings `data/esm2_{150m,650m}_embeddings/` (3244 each), datasets `dataset_esm2_{150m,650m}.parquet`. **RESULT (homology): ESM-2 650M cls 0.960 / reg R² 0.181; 150M 0.920/0.162 — beats best DNA-LM (0.727/0.077) AND AA-composition (0.735/0.090).** Re-run probes: `uv run scripts/rerun_on_split.py --only-esm2`.

## Reframe brief for Austin
`docs/notes/tss_homology_collapse_brief.md` — 3-part story (CDS=composition, TSS=leakage/noise, ESM-2 wins) + skip-masking rationale + decisions for Austin. **Hold manuscript (#10) for his steer.**

## DONE this session (committed)
- ESM-2 paired-bootstrap CIs (`03063a9`): all 6 ESM-2 vs AA/DNA/scaling CIs exclude 0.
- Split-seed sensitivity (`cd6f3ad`): `scripts/seed_sensitivity.py`, seeds 42/1/7/123, all 3 conclusions stable (see brief). Per-seed metrics `data/seed_sensitivity/`.
- **GOTCHA:** `train_logistic_probe.py` writes `data/confusion_5way_{source}.json` to a FIXED path — off-split probing clobbers the canonical homology matrices; restore with `git checkout -- data/confusion_5way_*.json` (done this session).

## Pending (mapped to MINA.md issues)

**Doable now — CPU, no Austin (highest-value quick wins first):**
1. **α-selection sensitivity table (MINA #3)** — show cls/reg conclusions don't flip when Ridge α is picked by cosine vs R². `train_baseline.py` already records both per-α (`alpha_sweep`); build a small table from existing runs or a focused re-run. NOT built.
2. **CDS-vs-TSS within-encoder paired test (MINA #10)** — `bootstrap_metrics.json` has DNA-LM-vs-composition and ESM-vs-X pairs but NOT "CDS vs TSS within each encoder". Add pairs (e.g. `dnabert2_meanD` vs `tss_dnabert2_meanmean`) to `bootstrap_test_uncertainty.py` `PAIRED_*` and run `--paired`. Quick.
3. **Supplementary 70%-identity split (MINA #1)** — `splits_homology70.json` EXISTS but was never probed. Re-probe the matrix on it (swap into `splits.json` like `seed_sensitivity.py` does, or add `--splits` plumbing). ⚠️ will clobber `data/confusion_5way_*.json` — restore after via `git checkout`.

**Writeup (fold into manuscript text):**
4. **Pre-specified pooling-selection rule (MINA #10)** — already enforced in code (val-only `sweep_C`/`sweep_alpha`); just state it explicitly.
5. **De-emphasize best-cell reporting (MINA #10)** — framing.

**Needs Austin (critical path):**
6. **Send Austin the brief** `docs/notes/tss_homology_collapse_brief.md` — gates everything below.
7. **#10 manuscript** (`dna_to_text_paper` submodule): regenerate tables/figures, register `kmer6/codon/aa1-3/gc` + `esm2_{150m,650m}` display rows, reframe title/abstract/claims per the 3-part finding (CDS=composition, TSS=leakage, ESM-2 wins). **Hold for Austin's steer.**

## MINA.md issue status (one-line each)
- #1 homology split: DONE (primary 40%); 70% supplementary split file exists but UNPROBED (pending item 3).
- #2 stronger baselines: DONE (composition + ESM-2 protein LM); manuscript display rows pending (item 7).
- #3 Ridge α by R²: DONE; sensitivity table pending (item 1).
- #4 TSS coding leakage: DONE (#4a quantify; #4b mask SKIPPED by consensus; homology collapse answers it).
- #10 paired CIs + seed sensitivity: DONE; CDS-vs-TSS pair (item 2) + pooling-rule/best-cell text (items 4–5) pending.

## Environment / gotchas
- Run everything via `uv run`. GPU: RTX 5060, 8 GB.
- `mmseqs` at `~/.local/bin/mmseqs` (static avx2). GTF at `data/annotation/` (gitignored). `fair-esm==2.0.0` in venv.
- **`data/dataset.parquet` is ABSENT** — `splits.loader.resolve_dataset_path()` falls back to `dataset_dnabert2_meanmean.parquet` for y/meta (all encoder parquets share `{ensembl_id,x,y,symbol,family,summary}`; y/meta identical across them).
- `splits.json` is NOW the homology split. `metrics_homology.json` = homology results; `metrics.json` = old random (preserved). Confusion matrices now reflect homology.
- **GateGuard hook**: a fact-forcing gate blocks the first Bash and every Edit/Write — present the 4 requested facts, then retry the identical call. To disable for an implementation session: `ECC_GATEGUARD=off` or add `pre:edit-write:gateguard-fact-force` (and `pre:bash:gateguard-fact-force`) to `ECC_DISABLED_HOOKS`.
- 24 `data/dataset_tss_*.parquet` present (untracked) for the TSS arm / #8.
