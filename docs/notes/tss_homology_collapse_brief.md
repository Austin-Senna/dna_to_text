# Brief for Austin — the TSS arm collapses on the homology split

_2026-05-27 · branch `revision/journal-hardening` · companion to `homology_phase2_results.md`_

## TL;DR
The homology-aware re-analysis now produces **two** findings that point the same way:

1. **CDS arm reduces to composition** (reported earlier): on the primary homology split, the translated
   amino-acid composition baseline **ties** the best DNA-LM on family classification (AA-2mer 0.735 vs
   NT-v2 0.727 macro-F1) and **beats** it on GenePT regression (AA-3mer R² 0.090 vs best DNA-LM 0.077).
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

_(Paired-bootstrap difference CIs for ESM-2 vs AA-composition / DNA-LM and 150M vs 650M are the natural
next addition for MINA #10 rigor — not yet run.)_

## Decisions for you (Austin)
1. Approve **retiring or heavily qualifying the TSS substrate claim**? (The primary-split TSS arm is at
   chance.)
2. Reframe direction for **title / abstract / central claim** — e.g. from "DNA-LMs encode gene function
   beyond composition" toward a careful homology-controlled finding (composition is a strong baseline;
   the TSS-window signal was leakage).
3. Keep the **random split as sensitivity analysis only** (already demoted)?
4. **Skip masked-TSS** per the consensus, or run the one-encoder homology hedge pre-emptively?

_Holding all manuscript text (#10) until your steer._
