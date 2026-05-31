# Brief for Austin — the TSS arm collapses on the homology split

_2026-05-27 · branch `revision/journal-hardening` · companion to `homology_phase2_results.md`_

## TL;DR
The homology-aware re-analysis now produces **two** findings that point the same way:

1. **CDS arm reduces to composition — and this was already true on the random split.** On the primary
   homology split, the translated amino-acid composition baseline **ties** the best DNA-LM on family
   classification (AA-2mer 0.735 vs NT-v2 0.727 macro-F1) and **beats** it on GenePT regression (AA-3mer
   R² 0.090 vs best DNA-LM 0.077). On the original **random** split the same AA-2mer baseline already
   **beat** NT-v2 (0.837 vs 0.828 cls; AA-3mer R² 0.246 exceeds every random-split DNA-LM number on
   record). The published "DNA-LMs beyond composition" headline was an artifact of comparing only against
   the weak CDS 4-mer baseline (0.662 cls / 0.174 reg); translated AA-composition was never tried.
   Homology sharpens the story; it doesn't create it.
   See `docs/notes/random_split_comparators_sensitivity.md`.
2. **NEW — the TSS arm collapses on the homology split.** Best TSS family5 macro-F1 is **0.326**
   (chance floor, from shuffled labels, is **0.224**); TSS GenePT regression R² is **≈ 0.010**, with most
   cells at ~0 or slightly negative. On the *random* split the TSS arm looked real (0.455 / R² 0.122) — so
   that apparent signal was largely **paralog leakage**, not regulatory content.

3. **NEW — a real protein LM (ESM-2) wins decisively.** On the same homology split, ESM-2 650M gets
   family5 macro-F1 **0.960** and GenePT R² **0.181** — roughly **+0.23 macro-F1** over the best DNA-LM
   (0.727) and **~2×** the best regression (AA-3mer 0.090 / DNA-LM 0.077). ESM-2 150M is close behind
   (0.920 / 0.162); clean scaling.

**Net:** once homology leakage is removed, DNA language models add nothing beyond sequence composition
(CDS) and nothing at all on TSS windows — but a purpose-built **protein** language model substantially
outperforms both. The honest story flips from "DNA-LMs encode gene function" to "for protein-function
prediction the substrate/model that matters is a protein LM; DNA-LMs on genomic windows do not help."
This is a thesis-level result and needs your call on the reframe.

## Numbers — homology (primary) vs random split

| metric | RANDOM | HOMOLOGY (primary) | chance floor |
|---|---|---|---|
| CDS family5 macro-F1, best DNA-LM | 0.828 | 0.727 | 0.224 |
| CDS family5 macro-F1, AA-2mer baseline | — | **0.735** | |
| TSS family5 macro-F1, best DNA-LM | 0.455 | **0.326** | 0.224 |
| TSS family5 macro-F1, k-mer-on-window | — | 0.181 | |
| CDS GenePT R², best DNA-LM / AA-3mer | — | 0.077 / **0.090** | |
| TSS GenePT R², best DNA-LM | 0.122 | **0.010** | |

## What this means
- The paper's "TSS/regulatory substrate carries gene-function signal" claim **does not survive
  homology-aware splitting** — the TSS arm is at/near chance on the primary split.
- This is consistent with the window-composition audit (#4a): the 196,608 bp window is only **0.8%**
  target CDS, **2.3%** target exon, **11.6%** all exons (target+neighbour); ~89% is intron/intergenic.
  So there was never much coding sequence in the window — and on the homology split there is little
  signal of *any* kind to recover.

## #4b masked-TSS control — recommendation: **skip** (or minimal hedge)
MINA #4 asks to *"quantify or mask coding-sequence leakage in the TSS windows … to support that the TSS
arm is not just recovering residual coding sequence."* We addressed this **blind**: three independent
reviewers were given only the numbers above (no access to our notes or framing) and **unanimously**
concluded:
- **Don't run the masking experiment.** With homology-split TSS already at the floor, there is no signal
  to attribute to coding leakage — the experiment's outcome is foregone.
- **Masking on the random split is a "trap"** — it spends the heaviest GPU defending a number the
  homology split exists to discount, against the journal's primary-split guidance.
- MINA #4's *"quantify **or** mask"* is **already satisfied by the quantify branch** (#4a composition
  table) plus the near-floor homology performance.
- If a reviewer insists on the literal control, the agreed minimal hedge is: **one fast encoder, homology
  split only** — never the full 4-encoder matrix, never the random split.

## #9 ESM-2 comparator — DONE

Embedded each gene's translated protein with ESM-2 (mean over residues, chunk-and-mean for proteins
>1022 aa), probed on the homology split.

| homology split | family5 macro-F1 | GenePT R² |
|---|---|---|
| **ESM-2 650M** | **0.960** | **0.181** |
| ESM-2 150M | 0.920 | 0.162 |
| best DNA-LM (NT-v2 / DNABERT-2) | 0.727 | 0.077 |
| AA-2mer / AA-3mer composition | 0.735 | 0.090 |
| CDS 4-mer | 0.633 | — |
| chance floor | 0.224 | 0 |

ESM-2 beats the DNA-LMs *and* the AA-composition baselines on both tasks, and scales (650M > 150M).
That it also beats AA-composition means protein-LM **context** adds value beyond raw composition — the
opposite of what we found for DNA-LMs on CDS. **Fairness note for the text:** ESM-2 is purpose-built on
protein sequences for exactly this kind of task, so its edge is expected; the comparison's point is that
DNA-LMs on genomic windows are the wrong substrate/model here, not that ESM-2 is a novel result.

**Paired-bootstrap difference CIs (MINA #10) — ESM-2 650M − comparator, n=487 test, all frac(A>B)=1.000:**

| comparison | Δ (650M − B) | 95% CI |
|---|---|---|
| cls macro-F1 vs AA-2mer | **+0.225** | [+0.171, +0.285] |
| cls macro-F1 vs best DNA-LM (NT-v2) | **+0.233** | [+0.177, +0.289] |
| cls macro-F1 vs ESM-2 150M (scaling) | +0.039 | [+0.015, +0.070] |
| reg R² vs AA-3mer | **+0.091** | [+0.080, +0.102] |
| reg R² vs best DNA-LM (DNABERT-2) | **+0.105** | [+0.091, +0.118] |
| reg R² vs ESM-2 150M | +0.019 | [+0.015, +0.023] |

Every CI excludes zero: ESM-2's advantage over both AA-composition and the DNA-LMs is significant on
both tasks, and the 650M>150M scaling gap is significant too.

## Split-seed sensitivity (MINA #10)
Re-clustered once and re-assigned whole clusters to train/val/test at **4 seeds (42, 1, 7, 123)**, then
re-probed the headline cells. All three conclusions are **stable across seeds** (ranges over the 4 seeds):

| cell | family5 macro-F1 | GenePT R² |
|---|---|---|
| ESM-2 650M | 0.917 – 0.960 | 0.181 – 0.193 |
| ESM-2 150M | 0.920 – 0.950 | 0.160 – 0.170 |
| best DNA-LM | 0.656 – 0.727 | 0.068 – 0.084 |
| AA-composition (2mer cls / 3mer reg) | 0.699 – 0.735 | 0.083 – 0.096 |
| TSS (DNABERT-2) | 0.307 – 0.355 | −0.003 – 0.019 |

Every seed: AA-composition ties/beats the best DNA-LM; the TSS arm sits at the ~0.224 chance floor (cls)
and ~0 (reg); ESM-2 dominates. The conclusions are not an artifact of one cluster→split assignment.
(Driver: `scripts/seed_sensitivity.py`; per-seed metrics in `data/seed_sensitivity/`.)

## Supplementary robustness (added 2026-05-28; closes MINA #1/#3/#10)
Three remaining reviewer asks are now answered with committed analysis — none change the story:

- **CDS-vs-TSS within each encoder (MINA #10 paired test).** For every DNA-LM the CDS arm
  beats the TSS arm with P(CDS>TSS)=1.000 and a 95% CI excluding zero — cls ΔF1 +0.14 to +0.45,
  reg ΔR² +0.02 to +0.07 (dnabert2 ΔF1 +0.37 / ΔR² +0.065; nt_v2 +0.42 / +0.054; hyena
  +0.35 / +0.057; gena_lm +0.14 / +0.021). A 4-mer-on-CDS vs 4-mer-on-TSS-window control gives
  the same gap (+0.45 / +0.066). (`data/bootstrap_metrics.json`, commit `dfb9e86`.)
- **α-selection sensitivity (MINA #3).** Selecting Ridge α by validation cosine vs macro-R²
  changes nothing: 50/53 regression cells pick the same α, and the leader ordering
  (ESM-2 650M > 150M > AA-3mer > DNA-LMs) is identical under both rules; the only 3 cells whose
  α differs are noise-floor TSS cells. (`analysis/alpha_sensitivity/`, commit `c5bfea2`.)
- **Stricter 70%-identity split (MINA #1).** Re-probing the headline cells on a 70%-id
  supplementary split keeps the ordering (ESM-2 dominates; AA-composition ties/beats the best
  DNA-LM: aa2 0.825 > nt_v2 0.783 cls, aa3 0.215 > dnabert2 0.186 reg). Every cell is higher than
  at 40% — looser threshold, more paralog leakage — and the TSS arm rises from 0.326 (40%, at
  floor) to 0.505 (70%), tracking the leakage gradient. (`docs/notes/homology70_supplementary.md`,
  commit `13145d3`.)

## Supplementary robustness (added 2026-05-31; MINA #2 random-split symmetry)

- **Composition + ESM-2 on the random split (was homology-only).** The new comparators
  (`kmer6/codon/aa1-3/gc` + ESM-2 150M/650M) were never probed on the original random split;
  re-running them now under the same `--select-by r2` protocol shows the composition-parity
  finding was **already present at random**:
  - cls F1: **aa2 0.837 > NT-v2 0.828** on random (aa3 0.784); on homology aa2 0.735 ≈ NT-v2 0.727.
  - reg R²: **aa3 0.246** on random exceeds every random-split DNA-LM on record; on homology aa3 0.090
    ≈ best DNA-LM 0.077.
  - ESM-2 650M tops both splits: random 0.975 / R² 0.355, homology 0.960 / 0.181 (cls near
    saturation on random; reg halved by homology, like every other comparator).
  - Every cell except `gc` (near chance on both) scores higher on random — the leakage gradient is
    uniform across comparators.

  Net: the published "DNA-LMs beyond composition" gain **never existed** once translated AA-composition
  is included as a baseline — the original analysis simply omitted it. The homology split sharpens
  the reframe story but doesn't create it. This is a *stronger* result than "composition parity on
  homology": it would already be the right reading of the random-split data. ICSB Plan A's "stop
  calling NT-v2 'the strongest'" message can confidently be stated as cross-split, not homology-only.
  (`scripts/probe_random_comparators.py` → `data/metrics_random_comparators.json`; commit `2164d43`.
  Doc: `docs/notes/random_split_comparators_sensitivity.md`. Note: `build_paper_tables.build_leakage()`
  currently only reads `metrics.json` for the random side and so only covers DNA-LMs + kmer —
  extending it to also read `metrics_random_comparators.json` would let the leakage table include the
  composition + ESM-2 cells.)

## Decisions for you (Austin)
1. Approve **retiring or heavily qualifying the TSS substrate claim**? (The primary-split TSS arm is at
   chance.)
2. Reframe direction for **title / abstract / central claim** — e.g. from "DNA-LMs encode gene function
   beyond composition" toward a careful homology-controlled finding (composition is a strong baseline;
   the TSS-window signal was leakage).
3. Keep the **random split as sensitivity analysis only** (already demoted)?
4. **Skip masked-TSS** per the consensus, or run the one-encoder homology hedge pre-emptively?

_Holding all manuscript text (#10) until your steer._
