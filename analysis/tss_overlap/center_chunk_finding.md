# E5 center-chunk-position check: verdict

Aug 15 2026. Resolves the one open confound on the gena_lm `centermean` outlier (the `n//2`
chunk-position question). Tokenizer-only diagnostic over all 3,244 genes per encoder; no model, no GPU, no network.
Source: `scripts/check_tss_center_chunk.py` → `center_chunk_offsets_<enc>.csv`, `center_chunk_summary.json`.

> **UPDATE Aug 15 — the confirmatory re-probe (last section) REVISES the verdict below.** Re-anchoring on
> the TSS chunk confirmed gena_lm AND surfaced a second encoder (nt_v2) with recoverable TSS-proximal
> family signal that the mislabeled `centermean` had hidden. The global-pooling collapse still stands, but
> "gena_lm is a lone exception" is now too strong. Read the diagnostic verdict first, then the re-probe.

## The confound, restated
`centermean` reduces each gene to the chunk at token-count index `n_chunks // 2`
(`pooling_aggregator.aggregate`). The worry: for gena_lm's variable-length BPE, `n//2`-by-count could
land on a genomic region **offset from the TSS**, and a different locus than the other encoders — making
the 0.411 lift a pooling-definition artifact rather than TSS biology.

## What the check found (n = 3,244 per encoder)

| encoder | window | median chunk bp-width | median signed Δbp (chunk center − TSS) | `n//2` chunk contains the TSS | centermean F1 (hom) vs global-pool |
|---|---|---|---|---|---|
| dnabert2 | 510 tok | ~1.5–2.9 kb | **+1191** | **52.4 %** | 0.284 vs 0.326 — no lift |
| gena_lm  | 510 tok | ~1.9–4.1 kb | **+1437** | **50.5 %** | **0.411 vs 0.244 — LIFT** |
| nt_v2    | 998 tok | ~6.0 kb     | +5562 | 0.03 % | 0.275 vs 0.313 — no lift |
| hyena_dna| 8192 tok| ~4.6–8.2 kb | +5632 | 0.03 % | 0.218 vs 0.287 — no lift |

Two facts fall out:

1. **`centermean` is NOT "the TSS" for any encoder.** The signed Δbp is systematically **positive**
   (downstream) for all four: the `n//2`-by-count chunk sits ~`max_tokens/2` tokens 3′ of the base-pair
   midpoint because chunks overlap (stride) and the final chunk aligns to the sequence end. The docstring
   claim that "the middle chunk is the region at the TSS" (`pooling_aggregator.py:16-18`) is inaccurate —
   it is a TSS-*downstream* chunk, offset by half a chunk-window.

2. **How far off is pure chunk geometry — bp-width = `max_tokens × bp/token`.** The two 510-token encoders
   (gena_lm, dnabert2) get narrow ~2–4 kb chunks that stay TSS-proximal (contain the TSS base in ~50 % of
   genes; median center ~1.2–1.4 kb 3′ of it). The two wide-window encoders (nt_v2 ~6 kb, hyena ~8 kb) get
   chunks whose center is ~5.6 kb downstream and that **never** contain the TSS (0.03 % = 1 gene).

## Verdict: the artifact hypothesis is REFUTED; the gena_lm outlier is real

The decisive control is **gena_lm vs dnabert2** — geometric twins (same 510/64 chunking, both TSS-proximal).
At full corpus dnabert2's center chunk is if anything *closer* to the TSS than gena_lm's (contains it 52.4 %
vs 50.5 %; median offset +1191 vs +1437 bp). If the 0.411 lift were "gena_lm happens to grab a TSS-proximal
chunk," dnabert2 — grabbing an equally/more TSS-proximal chunk of the same window — should lift at least as
much. It does not (0.284, below its own global pool). So the lift is **not** produced by where the chunk
lands; it is **specific to gena_lm's representation** of the TSS-proximal promoter region.

Therefore:
- The original confound ("gena_lm `n//2` is a random offset locus unlike the other encoders") is **false**.
  gena_lm's center chunk is reliably TSS-proximal (~0–4 kb 3′ of the TSS), geometrically near-identical to
  dnabert2's.
- The manuscript's **global-pooling TSS-collapse claim stands** — nothing here touches it.
- gena_lm + `centermean` is a **real, encoder-scoped exception**, safe to word as such.

## Two caveats the write-up must carry (honesty)
- **Relabel.** `centermean` is a *TSS-proximal downstream* chunk (~0–4 kb 3′ of the TSS for the 510-token
  encoders), not the TSS itself. Do not describe it as "the TSS chunk."
- **Not a controlled cross-encoder comparison.** The wide-window encoders' center chunk misses the TSS
  window entirely (0 % contain it), so "near chance for nt_v2/hyena_dna" is partly confounded by their chunk
  being off-TSS. The clean claim rests on the **matched gena_lm-vs-dnabert2 contrast**, not the 4-way sweep.

## Recommended confirmatory step (run below)
Re-anchor each encoder's chunk on the **TSS-containing** base pair (replace `n//2`-by-count with the chunk
whose bp span brackets the TSS) and re-probe family5. This (a) fixes the relabel and (b) gives nt_v2/hyena a
fair same-locus test. Prediction from this diagnostic: gena_lm's signal persists (its chunk is already
TSS-proximal); the other three stay near chance if the collapse is genuine. This is done in the confirmatory
re-probe section below (the cached per-chunk `mean` makes it CPU-only, no re-extraction).

## Provenance / reproduction
`uv run python scripts/check_tss_center_chunk.py` (all 4 encoders, full corpus). Windows + Ensembl lookups
read from `data/enformer_windows/` (cached Aug-14 E5 run). TSS bp offset recomputed per gene from the lookup
JSON via `enformer_windows.centered_window` (not assumed = length/2; chromosome-start-clipped genes handled).
Hand-verified on 3 gena_lm genes (chunk bp span, TSS offset, contains-flag all consistent).

---

# Confirmatory re-probe: TSS-anchored chunk pooling

Aug 15 2026. The diagnostic above showed `centermean` is a mislabeled TSS-*downstream* chunk and not a
same-locus comparison across encoders. This re-probes family5 on a **TSS-anchored** feature: per gene, the
chunk whose bp span is most TSS-centered (`argmin_k |chunk_bp_center − TSS|`, = `tss_chunk_idx`), pulled from
the already-cached per-chunk `mean` (no GPU, no re-extraction). Same logistic protocol as the 0.411 run
(C-sweep on val, refit train+val, eval test). Scope: all 4 encoders, homology + disjoint splits.
Sources: `scripts/build_tss_anchored_datasets.py`, `scripts/probe_tss_anchored.py`;
`data/dataset_tss_<enc>_tssanchored.parquet`, `data/metrics_tss_anchored{,_disjoint}.json`.

**Protocol sanity gate (passed):** re-running `tss_gena_lm_centermean` under this harness reproduced
**0.4111** exactly — the anchored numbers use an unchanged protocol.

## Results — family5 macro-F1 (chance 0.224)

Anchored columns show point [95% bootstrap CI], 1000× stratified test-set resamples
(`scripts/bootstrap_tss_anchored.py` → `data/bootstrap_tss_anchored.json`). **vs global** = whether the
anchored 95% CI clears the global-pool point (a signal-recovery test).

> **UPDATE Aug 15 — numbers below are the max_iter=5000 re-freeze.** The original grid used
> `max_iter=2000`, at which two cells were cut off before lbfgs converged (gena_lm hom needed 2059 iters,
> hyena_dna hom/disj needed 3808/4198). Re-freezing at `max_iter=5000` (every anchored cell now converges;
> 10000 gives bit-identical results) moved gena_lm hom 0.4306→0.4323 and, more, hyena_dna 0.261/0.233→0.285/0.284.
> **No conclusion changes** (hyena's converged point still CI-overlaps its global pool). Table refreshed.

| encoder | global hom | CM hom | **anchored hom [CI95]** | vs global | global disj | CM disj | **anchored disj [CI95]** | vs global |
|---|---|---|---|---|---|---|---|---|
| dnabert2  | 0.326 | 0.284 | **0.287** [0.249, 0.331] | overlaps | 0.254 | 0.263 | **0.265** [0.227, 0.305] | overlaps |
| nt_v2     | 0.313 | 0.275 | **0.352** [0.300, 0.406] | overlaps | 0.259 | 0.256 | **0.374** [0.320, 0.430] | **clears** |
| gena_lm   | 0.244 | 0.411 | **0.432** [0.393, 0.468] | **clears** | 0.206 | 0.393 | **0.414** [0.391, 0.439] | **clears** |
| hyena_dna | 0.287 | 0.218 | **0.285** [0.238, 0.335] | overlaps | 0.265 | 0.192 | **0.284** [0.240, 0.330] | overlaps |

(Global-pool numbers = each encoder's **best** global pool per split — the *hardest* bar to clear, a
conservative choice — named per cell: homology dnabert2/nt_v2 meanmean, gena_lm clsmean (0.244), hyena meanD;
disjoint dnabert2 meanmean, nt_v2 maxmean, gena_lm/hyena meanG. Sourced from **`metrics_homology.json`** and
**`metrics_tss_disjoint.json`** — NOT `metrics.json`, which holds the leaky random-split TSS numbers
(split-provenance trap). These global-pool bars are the **frozen accepted-paper numbers (max_iter=2000)** and
are NOT regenerated; only the E5-specific anchored/composition features are re-frozen at 5000. The `vs global`
column here is the coarse CI-vs-point test; the authoritative signal-recovery test is the **paired bootstrap**
in the paired-bootstrap section below. Anchored point estimates reproduce the probe's test macro-F1 exactly — same C,
split, deterministic lbfgs fit.)

## What it establishes

1. **gena_lm — exception confirmed and sharpened.** Anchoring exactly on the TSS chunk *raises* the signal
   (0.411→**0.431** hom, 0.393→**0.414** disj) and it survives the disjoint split. Sharpening the anchor
   sharpens the signal ⇒ it is genuinely TSS-localized, not a stray-chunk artifact. Still concentrated in
   kinase (0.95) + tf (0.66).

2. **dnabert2 — twin control HOLDS (the decisive result).** dnabert2 is gena_lm's geometric twin (same
   510/64 chunking, equally TSS-proximal center chunk). With the *identical* TSS anchor it stays at
   **0.287** hom / **0.266** disj — essentially its centermean value and still **below its own global pool
   0.326**. So a TSS-anchored chunk manufactures no family signal on its own; gena_lm's lift is
   encoder-specific, not a property of the locus or the pooling. This is the clean same-geometry,
   same-locus contrast the diagnostic promised.

3. **nt_v2 — a SECOND signal the mislabeled centermean hid (new, modest, split-dependent).** nt_v2's
   `centermean` chunk sat ~5.5 kb off-TSS (0% contained the TSS), so its 0.275 undersold it. With a fair TSS
   anchor it rises to **0.351** [0.299, 0.406] hom / **0.374** [0.320, 0.430] disj — signal over gpcr (0.39)
   + tf (0.71). **CI verdict:** both CIs clear *chance* (0.224) comfortably ⇒ a real above-chance TSS-proximal
   signal. But it clears its *global pool* only on the disjoint split (0.374 CI lower 0.320 > 0.259);
   on homology the CI [0.299, 0.406] straddles global 0.313, so anchored ≈ global there. Read: nt_v2 carries
   genuine above-chance TSS signal that the off-TSS centermean missed, but "beats global pooling" holds only
   under the leakage-controlled split — real but modest, not on gena_lm's footing. (The earlier "disjoint >
   homology, likely noise" worry is resolved: the two anchored CIs overlap heavily, so ~±0.03 across splits
   is within resampling noise; the CI, not a seed re-run, was the right uncertainty tool since lbfgs is
   deterministic on a fixed split.)

4. **hyena_dna — flat.** Anchored 0.261 hom / 0.233 disj, barely above chance (0.224); its centermean
   0.218/0.192 was slightly *below* chance because its 8 kb center chunk was fully off-TSS. No real recovery.

## Revised verdict (supersedes the diagnostic's "lone exception")

- **The global-pooling TSS-collapse HEADLINE still stands, untouched.** Under global pooling (the
  manuscript's TSS arm), disjoint macro-F1 is at the chance floor for all four (0.254 / 0.259 / 0.206 /
  0.265). Nothing here changes that.
- **But the stronger interpretive claim — "the TSS window carries no recoverable family signal for DNA
  encoders" — is now contradicted, with CI support.** A correctly TSS-anchored chunk clears global pooling
  for **gena_lm on both splits** (CI 0.392–0.465 hom, 0.391–0.439 disj; strong, split-consistent) and for
  **nt_v2 on the disjoint (leakage-controlled) split** (0.374, CI lower 0.320 > global 0.259; modest, and
  above chance on both splits though ≈ global on homology). dnabert2 (twin control) and hyena_dna stay
  CI-overlapping their global pool — nothing. So the signal is there, in the TSS-proximal region, for some
  encoders; global pooling dilutes it.
- The question this answers — *does global pooling hide a central-TSS signal?* — is settled: for some
  encoders, yes. The global-pooling collapse is not evidence of absent signal, only of dilution.

## Interpretation
- The collapse-under-global-pooling result is the robust headline (survives the disjoint split).
- TSS-anchored pooling clears global-pool family signal for **gena_lm** (both splits, strong) and **nt_v2**
  (disjoint only, modest), not for dnabert2/hyena_dna. "TSS pooling stays at chance" is the mislabeled-
  `centermean` artifact; this fair anchored test refutes it.
- `centermean` should be described as a TSS-proximal-*downstream* chunk, not "the TSS chunk"; the anchored
  pooling is the corrected same-locus TSS pooling.
- Accurate summary: **coarse-family collapse under global pooling, with encoder-specific TSS-proximal signal
  recoverable by TSS-anchored pooling (gena_lm strong on both splits; nt_v2 modest, disjoint only).** CIs are
  in `data/bootstrap_tss_anchored.json`; lbfgs is deterministic on a fixed split, so the test-set bootstrap
  (not a seed re-run) is the uncertainty source.

---

# Composition control: is the recovered signal biology or promoter composition? (Aug 15)

The decisive test the diagnostic couldn't answer: featurize the DNA of the **same anchored chunk** each
encoder pooled (4-mer + GC, and 6-mer) and probe family5. If sequence composition of that region matches the
encoder, the "encoder recovers TSS biology" claim is hollow. Scripts: `build_tss_composition_baseline.py`,
`probe_tss_composition.py` → `data/dataset_tss_<enc>_{chunk4mergc,chunk6mer}.parquet`,
`metrics_tss_composition{,_disjoint}.json`. CPU-only, cached windows, no GPU.

## Result — chunk composition macro-F1 (chance 0.224), vs the encoder embedding

| encoder | encoder anchored (hom/disj) | chunk 6-mer (hom/disj) | chunk 4-mer+GC (hom/disj) |
|---|---|---|---|
| gena_lm  | **0.431 / 0.414** | 0.209 / 0.206 | 0.212 / 0.222 |
| nt_v2    | 0.351 / 0.374 | 0.196 / 0.194 | 0.208 / 0.207 |
| dnabert2 | 0.287 / 0.265 | 0.208 / 0.212 | 0.239 / 0.223 |
| hyena_dna| 0.261 / 0.233 | 0.190 / 0.198 | 0.193 / 0.222 |

Composition is **at chance (0.19–0.24) for every encoder's chunk on both splits.** Per-class, every
composition cell has the same shape: **tf 0.89–0.94, kinase 0.00–0.04, gpcr ~0.25, immune/ion ≈ 0.**

## What it establishes (the sharp claim)
1. **The naive confound is refuted.** k-mer/GC of gena_lm's own anchored chunk scores 0.21, not 0.43 — the
   encoder's 0.431 is NOT explained by the chunk's sequence composition.
2. **TF-at-TSS is pure promoter composition — definitively.** Composition recovers tf at **0.90–0.92** (higher
   than the encoders' tf), across all 8 cells. This is why every encoder's macro-F1 looks non-trivial: the tf
   class is a k-mer effect (TF genes = canonical CpG-island promoters), not encoder skill. Consistent with the
   manuscript's own hedge that the TSS window carries "at most indirect promoter or cis-regulatory cues."
3. **Kinase is never compositional.** Composition gets kinase **0.00–0.04** everywhere (both splits, all
   encoders). So the kinase family carries no k-mer/GC signal in the TSS-proximal region.
4. **gena_lm's kinase recovery (0.95 hom / 1.00 disj) is therefore a genuine, encoder-specific, NON-
   compositional TSS-proximal signal — the only such signal in the whole experiment.** The twin control is now
   airtight: gena_lm and dnabert2 have identical composition profiles (~0.21, tf-driven, kinase 0) and identical
   chunk geometry, yet gena_lm's *encoder* adds kinase 0.95 while dnabert2's adds nothing (0.287 ≈ its
   composition 0.24).
5. **nt_v2 is weak-but-real, not composition.** Its encoder (0.35) beats its composition (0.20) via diffuse
   gains in the classes composition scores zero (kinase 0.25, immune 0.17, ion 0.17) — so the signal is
   non-compositional, but it does not beat *global pooling* except on the disjoint split. Report as suggestive.

**Final framing:** the TSS-proximal region carries family signal in two separable pieces — a **TF component
that is trivial promoter composition** (any k-mer gets it; present in all encoders and in the wide-window
baseline) and a **kinase component that only gena_lm's encoder recovers and that sequence composition cannot**
(0.94 vs 0.02–0.06). The global-pooling collapse headline stands; the honest positive result is narrow and
specific: *gena_lm uniquely encodes a non-compositional kinase-promoter signal at the TSS.* This closes the
composition confound: gena_lm anchored 0.432 exceeds both the 0.326 best global-pool macro-F1 and the 0.401
supervised-Enformer comparator, and it rests on real, non-artifactual (encoder-specific, non-compositional)
signal.

## Safety / provenance
No GPU, no re-extraction (cached per-chunk `mean`; composition from cached windows). No commit; HEAD `fb06393`.
Tracked `data/metrics.json`
and `data/splits.json` untouched; the disjoint pass backed up + restored `splits.json` and all
`confusion_5way_*.json` in a `finally` (verified: `splits.json` unmodified post-run). New untracked artifacts
only (`dataset_tss_<enc>_tssanchored.parquet`, `confusion_5way_tss_<enc>_tssanchored.json` [homology],
`metrics_tss_anchored{,_disjoint}.json`, `bootstrap_tss_anchored.json`) — post-Aug-17 hygiene like the rest
of E5. The bootstrap run re-verified `splits.json` unmodified after its own disjoint swap.

---

# Result hardening (Aug 15): paired test, anchor robustness, per-class, max_iter=5000 re-freeze

Hardening pass for the TSS-anchored result: a paired significance test, an anchor-rule robustness check, a
per-class decomposition, and a convergence re-freeze. All CPU, no GPU; tracked
`data/metrics.json`/`data/splits.json` untouched (verified: `splits.json` byte-identical to HEAD after every
disjoint swap; `metrics.json` mtime predates these runs). Numbers here are the converged `max_iter=5000` grid.

## Paired bootstrap — the authoritative signal-recovery test
The `vs global` column in the results table is CI-vs-point: it compares the anchored CI lower bound to a
*scalar* global-pool value, ignoring the baseline's own uncertainty and unpaired. The repo's paired difference
bootstrap (`bootstrap_test_uncertainty.paired_bootstrap_classification`, used for E2 and the CDS-vs-TSS rows)
resamples the SAME test genes once per iteration and applies to both fixed predictions → Δmacro-F1 [CI] +
P(anchored>global). Anchored and global-pool parquets are row-identical (3244), so pairing is exact. Both sides
refit at `max_iter=5000`. Source: `scripts/paired_tss_anchored.py` → `data/paired_tss_anchored.json`.

| encoder | vs best global pool | **Δ hom [CI95], P(anc>gl)** | **Δ disj [CI95], P(anc>gl)** | clears global? |
|---|---|---|---|---|
| gena_lm   | clsmean / meanG   | **+0.161 [+0.105,+0.215], 1.000** | **+0.205 [+0.162,+0.246], 1.000** | **both splits** |
| nt_v2     | meanmean / maxmean| +0.047 [−0.021,+0.111], 0.906     | **+0.103 [+0.030,+0.177], 0.995** | **disjoint only** |
| dnabert2  | meanmean / meanmean| −0.039 [−0.102,+0.023], 0.111    | +0.012 [−0.046,+0.068], 0.654     | neither (twin control) |
| hyena_dna | meanD / meanG     | +0.031 [−0.030,+0.088], 0.834     | +0.018 [−0.040,+0.074], 0.724     | neither (flat) |

"Clears global" = Δ CI excludes 0 (equivalently P ≥ 0.975). The paired test **confirms the CI-vs-point verdict
and sharpens it**: gena_lm clears on both splits (P=1.000); nt_v2 clears on the leakage-controlled disjoint
split only (hom P=0.906, CI straddles 0); dnabert2 and hyena_dna clear on neither. Note hyena's *converged*
anchored point (0.285/0.284) is higher than its cut-off 2000-era value, but the paired test — refitting the
global side at 5000 too — still finds no difference. The paired difference bootstrap is the same test used
elsewhere in this analysis for the CDS-vs-TSS and composition comparisons.

## nt_v2 is suggestive, not "clears global"
Across the **8 encoder×split cells**, exactly two are positive under the paired test: gena_lm (both) and
nt_v2 (disjoint only). nt_v2 is **1 positive cell of its 2** and its homology match is null (P=0.906) — it will
not survive multiple-comparison correction. **Only gena_lm carries a "clears global pooling" claim.** Report
nt_v2 as *suggestive: above chance on both splits (both anchored CIs clear 0.224), non-compositional (see the
composition control), but it beats its own global baseline only on the disjoint split.* The
`ci_clears_global_pool` boolean in `bootstrap_tss_anchored.json` is raw per-cell data, NOT a
multiplicity-corrected claim — do not quote it as "nt_v2 clears global" without the disjoint-only caveat.

## Anchor robustness — the anchored chunk already contains the TSS (rule is invariant, not merely robust)
The audit worried that "TSS-anchored" (= `argmin_k |chunk_center − TSS|`) is nearest-center, not TSS-containing,
and that for nt_v2/hyena the chunk "never contains the TSS (0%)." That 0% is the **`centermean` `n//2`-by-token
chunk**, which sits downstream — NOT the anchored feature. Extending the geometry diagnostic to flag containment
(`scripts/check_tss_center_chunk.py` → `tss_contain_chunk_idx`, `anchor_contains_tss`;
`center_chunk_offsets_<enc>.csv` cols added, existing cols byte-identical) shows:

| encoder | centermean `n//2` contains TSS | **anchored (argmin-center) contains TSS** | contains-rule picks a *different* chunk |
|---|---|---|---|
| dnabert2  | 52.4% | **100.0%** | 0 / 3244 |
| nt_v2     |  0.0% | **100.0%** | 0 / 3244 |
| gena_lm   | 50.5% | **100.0%** | 0 / 3244 |
| hyena_dna |  0.0% | **100.0%** | 0 / 3244 |

The anchored chunk brackets the TSS base pair for **every gene, every encoder** (chunks tile the window, and the
center-nearest chunk is always a containing chunk). An explicit contains-TSS anchor (`build_tss_anchored_datasets.py
--anchor containstss`) therefore selects the **identical** chunk on all 3244 genes — the `tsscontain` feature
would be byte-identical to `tssanchored`. So the anchor rule is not just robust, it is **invariant**: there is no
alternative same-locus rule to test, and building/probing `tsscontain` is a provable no-op (not run — the
0/3244 diff is the proof). This *strengthens* the result and retires the audit's premise, which conflated
`centermean` with the anchored feature. Honest scope for the write-up: `centermean`'s downstream chunk missed
the TSS for the wide-window encoders (0%), but the corrected **anchored** chunk contains it for all four.

## Per-class decomposition (mechanism, not the macro-F1 headline)
Per-class F1 (homology, 5000; from the frozen confusion matrices) decomposes the macro-F1 into two separable
pieces:

- **TF is universal and compositional.** tf F1 ≈ 0.68–0.71 for *all four* anchored encoders AND for chunk
  composition (4-mer+GC / 6-mer of the same chunk) — tf is the largest class and a canonical CpG-island
  promoter family, so any k-mer recovers it (composition tf recall ~0.9; the finding doc's earlier "tf 0.90–0.92"
  was per-class recall/accuracy, F1 ≈ 0.71). This is promoter composition, not encoder skill.
- **Kinase is gena_lm's alone and non-compositional.** kinase F1: gena_lm anchored **0.94**, dnabert2 (twin)
  **0.25**, the other encoders ~0.13–0.25; chunk composition **0.02–0.06** for every encoder. So gena_lm's
  encoder uniquely separates kinases, and sequence composition cannot.
- gpcr/immune/ion are near-collapsed for everyone (F1 ≤ 0.32).

**Reframe (narrower, sharper than the macro-F1):** *a TSS-proximal chunk recovers TF-family signal universally
(all encoders; promoter composition), and gena_lm's encoder uniquely also separates kinases (0.94 vs
composition 0.02–0.06) — the only encoder-specific, non-compositional TSS signal in the experiment.* The
global-pooling collapse headline is untouched.

## Provenance (this hardening pass)
`max_iter` bumped 2000→5000 in `src/linear_trainer/logistic_probe.py` (the ONE code edit in this pass).
Re-frozen at 5000: `metrics_tss_anchored{,_disjoint}.json`,
`confusion_5way_tss_<enc>_tssanchored.json`, `bootstrap_tss_anchored.json`, `paired_tss_anchored.json`,
`metrics_tss_composition{,_disjoint}.json` (composition macro-F1 unchanged — those fits already converged).
`bootstrap_tss_anchored.py` and `paired_tss_anchored.py` now READ the anchored best C from the probe metrics
(removed the hard-coded 2000-era C, a staleness trap). The global-pool bars stay frozen at the accepted-paper
2000 values. New anchor-robustness artifacts: `tss_contain_chunk_idx`/`anchor_contains_tss` columns + summary fractions
(`center_chunk_summary.json`); no `tsscontain` parquets built (invariant, 0/3244 diff).

---

# Scripts & reproduction

The full E5 chain and the artifacts it produces. Run order is `scripts/run_e5.sh` (steps 2–8; CPU-only,
no GPU). Step 1 is the GPU-gated upstream extraction and is a prerequisite, not part of the driver.
The chain never writes tracked `data/metrics.json` (mixed-provenance trap) — it writes its own JSONs.

| # | Script | Purpose | Reads → writes |
|---|---|---|---|
| 1 | `scripts/run_tss_extract_capped.py` | VRAM-capped launcher for `run_tss_multi_pool_extract.py`; caps GPU memory so an OOM raises a catchable error (resume from cache) instead of hanging WSL. **GPU, prerequisite.** | model → `data/tss_chunk_reductions_<enc>/*.npz` (gitignored, GPU-gated) |
| 2 | `scripts/check_tss_center_chunk.py` | Tokenizer-only diagnostic: where the `n//2` centermean chunk lands in bp vs the TSS + TSS-containment flags. | tokenizer → `analysis/tss_overlap/center_chunk_offsets_<enc>.csv`, `center_chunk_summary.json` |
| 3 | `scripts/build_tss_anchored_datasets.py` | Re-select, per gene, the truly TSS-centered cached chunk vector (`--anchor`); no GPU. | `.npz` cache → `data/dataset_tss_<enc>_tssanchored.parquet` |
| 4 | `scripts/probe_tss_anchored.py` | Family5 logistic protocol on the anchored datasets, both splits (safe `splits.json` swap/restore). | anchored parquets → `data/metrics_tss_anchored{,_disjoint}.json`, `confusion_5way_tss_<enc>_tssanchored.json` |
| 5 | `scripts/build_tss_composition_baseline.py` | Featurize the SAME anchored chunk's DNA into 4-mer+GC and 6-mer composition baselines. | anchored parquets → `data/dataset_tss_<enc>_{chunk4mergc,chunk6mer}.parquet` |
| 6 | `scripts/probe_tss_composition.py` | Same family5 protocol on the composition baselines, both splits. | composition parquets → `data/metrics_tss_composition{,_disjoint}.json`, `confusion_5way_tss_<enc>_{chunk4mergc,chunk6mer}.json` |
| 7 | `scripts/bootstrap_tss_anchored.py` | Stratified 1000× test-set 95% CIs on the anchored macro-F1 (reads best C from step 4). | anchored parquets + metrics → `data/bootstrap_tss_anchored.json` |
| 8 | `scripts/paired_tss_anchored.py` | Paired difference bootstrap: anchored vs each encoder's best global pool (Δmacro-F1 [CI] + P(anc>global)). | anchored parquets + metrics → `data/paired_tss_anchored.json` |

Code edits behind the chain: `src/data_loader/pooling_aggregator.py` (adds the `centermean` variant),
`src/data_loader/multi_pool.py` (`collect=False` to avoid accumulating GB of reductions in RAM),
`scripts/run_tss_multi_pool_extract.py` (passes `collect=False`), `scripts/train_logistic_probe.py`
(registers the `tss_<enc>_tssanchored` dataset names), `src/linear_trainer/logistic_probe.py`
(`max_iter` 2000→5000). These feed the paper's E5 supplementary table `s_tss_anchored.tex` (GAP 9) and
the anchored grid in `revision_plan.md`.
