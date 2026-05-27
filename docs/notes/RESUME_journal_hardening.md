# RESUME — journal-hardening phase (handoff)

Last updated: 2026-05-26. Branch: **`revision/journal-hardening`** (4 commits, working tree clean, not pushed).

## Read these first
- `docs/journal_hardening_plan.md` — the approved plan (5 MINA.md issues, phases, decisions).
- `docs/notes/homology_phase2_results.md` — the headline Phase-2 findings + paired CIs (the brief for Austin).
- Memory: `~/.claude/projects/-home-hayden-dna-to-text/memory/project_journal_hardening.md`.

## Locked decisions
Homology split = PRIMARY; both GPU optionals (#4b masked-TSS, ESM-2) in scope; MMseqs2 @40%+70%.

## Done (7/10) — commits
- `595ac07` Phase-1 code: #3 Ridge α→macro-R² (`probe.py`/`train_probe.py`), `src/protein/translate.py`, composition baselines (`kmer_baseline.featurize_kmer`, `src/composition_baseline/{codon,aa_kmer,gc}`), #10 paired bootstrap.
- `1edd3a9` fair-esm dep + gitignore (tools/, data/annotation/, data/cluster_work/).
- `43cb4c4` #1 homology split: `src/cluster/mmseqs_cluster.py`, `build_cluster_splits` in `src/splits/make_splits.py`, rewritten `scripts/make_splits.py`. Wrote `splits.json` (PRIMARY 40%), `splits_homology70.json`, `splits_random.json` (+`splits_random_original.json` backup).
- `d41fd4f` Phase 2: `scripts/rerun_on_split.py` (125 cells → `data/metrics_homology.json`), bootstrap synthetic sources + homology PAIRED lists, regenerated confusion matrices.

## KEY FINDING (drives the manuscript reframe — Austin's call)
Homology 40% split: NT-v2 beats 4-mer (ΔF1 +0.089 [+0.025,+0.161]) but is **tied with AA-2mer** (ΔF1 −0.013 [−0.070,+0.053]); on regression **AA-3mer beats the best DNA-LM** (ΔR² −0.014 [−0.021,−0.007]). "DNA-LMs beyond composition" does not survive. Cosine ≈0.92 for all cells (R² is the discriminating metric).

## #4a GTF overlap — DONE (code uncommitted)
Built `src/tss_overlap/{gtf,overlap}.py` + `scripts/tss_overlap.py` (numpy boolean-mask, no new dep). Ran over all 3,244 genes (0 skipped); partition buckets sum to 1.0 (asserted). **HEADLINE: the 196,608 bp window is overwhelmingly non-coding context — mean target-CDS = 0.0080 (0.8%), target-intron 0.17, neighbour-intron 0.39, intergenic 0.33.** Reinforces the AA-composition finding (DNA-LMs see ~1% coding). Outputs: `analysis/tss_overlap/{tables,figures}/` (per_gene_overlap.csv, overlap_by_family.{csv,md}, partition_by_family.png, intergenic_intron_distribution.png) — artifacts not committed. GTF cache `data/annotation/gtf_features.parquet` (gitignored). Re-run: `uv run scripts/tss_overlap.py`. **TODO: commit `src/tss_overlap/` + `scripts/tss_overlap.py`.**

## Pending (2 tasks) — exact next steps
- **#4b masked-TSS** (GPU, RTX 5060 8GB): GTF-driven mask CDS/exon→N in TSS windows, re-encode via `src/data_loader/multi_pool.py`+`model_registry.py` → `dataset_tss_*_masked*.parquet`, re-probe, masked-vs-unmasked deltas. **This also re-runs the TSS arm on the homology split** (TSS not yet re-run — extend `rerun_on_split.py` with `--tss`).
- **#9 ESM-2 comparator** (GPU): install present (`fair-esm`); run small ESM-2 on translated proteins (reuse `protein.translate_cds`), cache embeddings, probe family + GenePT-regression. Now extra-interesting: does a real protein LM beat the AA-composition baseline?
- **#10 manuscript** (`dna_to_text_paper` submodule): regenerate tables/figures via `scripts/build_analysis_artifacts.py` (also register baselines kmer6/codon/aa1-3/gc as display rows — deferred from #2). Reframe title/abstract/claims per the homology+AA finding. **Hold for Austin.** Also: α-selection sensitivity table, pre-specified pooling rule, TSS overlap, masked-TSS, ESM.
- Also pending: split-seed sensitivity (run `make_splits.py --seed N` for a few seeds + `rerun_on_split.py`).

## Environment / gotchas
- Run everything via `uv run`. GPU: RTX 5060, 8 GB.
- `mmseqs` at `~/.local/bin/mmseqs` (static avx2). GTF at `data/annotation/` (gitignored). `fair-esm==2.0.0` in venv.
- **`data/dataset.parquet` is ABSENT** — `splits.loader.resolve_dataset_path()` falls back to `dataset_dnabert2_meanmean.parquet` for y/meta (all encoder parquets share `{ensembl_id,x,y,symbol,family,summary}`; y/meta identical across them).
- `splits.json` is NOW the homology split. `metrics_homology.json` = homology results; `metrics.json` = old random (preserved). Confusion matrices now reflect homology.
- **GateGuard hook**: a fact-forcing gate blocks the first Bash and every Edit/Write — present the 4 requested facts, then retry the identical call. To disable for an implementation session: `ECC_GATEGUARD=off` or add `pre:edit-write:gateguard-fact-force` (and `pre:bash:gateguard-fact-force`) to `ECC_DISABLED_HOOKS`.
- 24 `data/dataset_tss_*.parquet` present (untracked) for the TSS arm / #8.
