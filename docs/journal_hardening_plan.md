# Journal-hardening plan (pre-submission, Bioinformatics-tier)

Source issues: repo `MINA.md` (#1, #2, #3, #4, #10). Branch: `revision/journal-hardening`.

## Locked decisions (2026-05-26)
- **#1 homology-aware split → PRIMARY result**; random split demoted to sensitivity. Headline NT-v2-vs-k-mer gain may shrink; accepted.
- **Both heavy GPU optionals in scope:** #4b masked-TSS re-encode control, #2 ESM-2 protein-LM comparator.
- **Clustering: MMseqs2 at 40% (primary) + 70% (supplementary)** identity/coverage.

## Key cost fact
Encoder embeddings are cached per-gene (`data/sequences/`, `chunk_reductions_*`, `data/dataset_*.parquet`) and are **split-independent**. Re-splitting + re-probing needs **zero DNA re-encoding** — only the cheap linear probes (logistic + Ridge) and bootstraps re-run. The only GPU re-encoding is #4b and ESM. GPU available: RTX 5060, 8 GB. Run everything via `uv run`.

## Per-issue summary
| # | Issue | Re-encode? | Effort | Compute |
|---|-------|-----------|--------|---------|
| 1 | Homology-aware cluster split (MMseqs2) — PRIMARY | No | High | Low (CPU probes) |
| 2 | Composition + translated baselines (+ESM) | No (ESM: yes) | Med | Low (ESM: med GPU) |
| 3 | Ridge α select by macro-R² not cosine | No | Trivial | Trivial |
| 4 | 4a GTF overlap summary; 4b masked-TSS control | 4a no / 4b yes | Med | 4a low / 4b high GPU |
| 10 | Paired difference CIs + pre-specified selection | No | Med | Low |

## Execution phases
- **Phase 0 — provision:** install MMseqs2, GTF tooling, `fair-esm`; download Ensembl GRCh38 GTF to `data/annotation/` (gitignored).
- **Phase 1 — code (CPU):** #3 α→R² (`src/linear_trainer/probe.py`, `scripts/train_probe.py`); shared `src/protein/translate.py`; #1 `src/cluster/mmseqs_cluster.py` + rewrite `src/splits/make_splits.py` (whole-cluster, family-balanced); #2 featurizers (`src/kmer_baseline` parametric k, `src/composition_baseline/{codon,aa_kmer,gc}`) registered in `scripts/train_logistic_probe.py`, `scripts/train_baseline.py`, `scripts/build_full_table.py`; #10 paired bootstrap in `scripts/bootstrap_test_uncertainty.py`; #4a `src/tss_overlap/overlap.py`.
- **Phase 2 — single big re-run on homology split (CPU):** regenerate `data/splits.json` (primary 40%), assert no cluster spans splits; re-run all classification + Ridge cells (R²-selected α) + all baselines → `data/metrics.json`; re-run bootstrap (per-cell + paired); split-seed sensitivity (40%×seeds + 70%); #4a summary.
- **Phase 3 — GPU tracks (parallel, resumable):** #4b mask CDS/exon→N, re-encode TSS windows, re-probe, masked-vs-unmasked deltas; ESM-2 on translated proteins as comparator row.
- **Phase 4 — manuscript (`dna_to_text_paper` submodule):** regenerate Tables 2–4, pooling heatmap, per-dim R²; update Results/abstract/title/claims to homology-aware primary; random split → Supplementary; appendix Ridge protocol + α-selection sensitivity, pre-specified pooling rule, TSS overlap, masked-TSS, ESM. Surface headline deltas to lead author (Austin) before paper text lands.

## Verification
- Split integrity: every cluster maps to one split; per-family proportions per split within tolerance; existing disjointness asserts pass.
- #3: `uv run python scripts/train_probe.py --dataset <cell>` prints val R² + cosine; α maximizes val R².
- #2: featurizer shapes (6-mer→4096, codon→64, AA-3mer→8000); probe runs and appears in `data/full_table.md`.
- #4a: overlap fractions per window sum ~1.0; spot-check a multi-exon gene.
- #4b: masked windows show N-runs over CDS coords; masked-vs-unmasked deltas reported with CIs.
- #10: paired difference CIs for NT-v2 meanD − 4-mer and CDS − TSS; split-seed spread reported.

## Progress
- DONE #3 (verified); shared translation (verified); #2 featurizers + classification wiring (verified, `gc` ran end-to-end, macro-F1 0.147 as expected weak baseline).
- REMAINING #2: regression-side registration (`scripts/train_baseline.py`) + `scripts/build_full_table.py` ROW_ORDER.
- PENDING: #10, #1, #4a, Phase 2, #4b, ESM, Phase 4.
