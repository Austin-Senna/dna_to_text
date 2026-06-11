# Random-split comparators (sensitivity for MINA #2)

_2026-05-31 · branch `revision/journal-hardening` · supplements the primary 40% homology split._

The journal-revision comparators (composition baselines `kmer6/codon/aa1-3/gc` plus
`esm2_{150m,650m}`) were added during hardening and only ever probed on the homology split.
For symmetry with the published random-split DNA-LM headline numbers (`data/metrics.json`),
this re-runs them on the original random 70/15/15 split under the same `--select-by r2`
protocol used for `data/metrics_homology.json`.

Driver: `scripts/probe_random_comparators.py` — swaps `splits_random.json` into
`splits.json`, runs the comparator cells, restores. Output:
`data/metrics_random_comparators.json` (legacy `metrics.json` and primary
`metrics_homology.json` untouched).

## Headline cells — random vs primary 40% homology

| cell | cls F1 (RAND) | cls F1 (HOM) | Δ | reg R² (RAND) | reg R² (HOM) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| **esm2_650m** | **0.975** | **0.960** | +0.015 | **+0.355** | **+0.181** | +0.174 |
| esm2_150m | 0.966 | 0.920 | +0.046 | +0.326 | +0.162 | +0.164 |
| **aa2** | **0.837** | **0.735** | +0.102 | +0.205 | +0.060 | +0.145 |
| aa3 | 0.784 | 0.655 | +0.129 | **+0.246** | **+0.090** | +0.155 |
| codon | 0.717 | 0.645 | +0.072 | +0.173 | +0.044 | +0.129 |
| aa1 | 0.696 | 0.681 | +0.015 | +0.165 | +0.031 | +0.134 |
| kmer (CDS 4-mer) | 0.662 | 0.633 | +0.029 | +0.174 | +0.045 | +0.129 |
| kmer6 (CDS 6-mer) | 0.579 | 0.546 | +0.032 | +0.185 | +0.057 | +0.128 |
| gc | 0.147 | 0.164 | −0.017 | +0.022 | −0.023 | +0.045 |

Reference DNA-LMs on the random split (from `data/metrics.json`, legacy α-by-cosine for
regression): `nt_v2_meanD` cls F1 **0.828**, `nt_v2_meanG` 0.826, `dnabert2_meanD` 0.738,
`dnabert2_meanG` 0.727.

## Reading

1. **AA-composition already beat NT-v2 on the random split.** AA-2mer cls F1 = **0.837**
   exceeds NT-v2 meanD's 0.828. AA-3mer reg R² = **0.246** exceeds every random-split
   DNA-LM regression number on record. The published "DNA-LMs encode gene function
   beyond composition" claim was an artifact of comparing only against the weak
   **CDS 4-mer** baseline (0.662 cls / 0.174 reg); had the original analysis included
   translated AA-composition, the headline gap would already have disappeared.
   The homology split **sharpens** this story but doesn't create it.

2. **ESM-2 dominates on both splits.** 650M random 0.975 / homology 0.960 (cls), 0.355 /
   0.181 (reg). Regression takes a bigger absolute hit from the homology control
   (R² halved for both ESM-2 sizes and for AA-3mer alike — a pattern shared with the
   DNA-LMs), but ESM-2's *relative* edge over DNA-LMs and AA-composition is preserved
   across splits. ESM-2 cls scores are already near saturation on the random split
   (0.975 / 0.966) so the homology gap is small (Δ +0.015 / +0.046).

3. **The leakage gradient is consistent across every comparator.** Every cell except
   `gc` (near chance on both) scores higher on the random split — the same direction
   as 40% vs 70% homology. The largest leakage drops are on regression
   (aa3 R² 0.246 → 0.090; esm2_650m 0.355 → 0.181), consistent with regression being
   the more discriminating metric (the manuscript already makes this point about
   cosine being compressed; here R² drops show the same).

**Net:** the random-split sensitivity run **strengthens** the reframe story rather than
weakening it. The "composition is a strong baseline" finding holds at *both* splits;
the published DNA-LM "beyond composition" gain never existed once translated
AA-composition is included as a baseline. ESM-2 still wins decisively at both splits.

(Does not change the TSS-arm collapse story: TSS cells weren't part of this run — those
continue to live in `metrics_homology.json` and `docs/notes/homology70_supplementary.md`.)
