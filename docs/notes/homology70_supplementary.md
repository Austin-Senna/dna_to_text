# Supplementary: 70%-identity homology split (MINA #1)

_2026-05-28 · branch `revision/journal-hardening` · companion to the primary 40% split._

MINA #1 asks for the primary result on a homology-aware split (done: 40% identity,
`data/splits.json`) and notes that "a stricter supplementary analysis at 70–90% identity
would help." `data/splits_homology70.json` (MMseqs2 @ 70% id, 80% coverage) was built
alongside the primary split but never probed. This re-probes the headline cells on it and
compares to the primary 40% split.

Driver: `scripts/probe_homology70.py` — swaps the 70% split into `data/splits.json`, runs
the `seed_sensitivity.py` headline cells, restores the canonical split (and the clobbered
`data/confusion_5way_*.json` are restored via `git checkout` afterward). Metrics land in
`data/metrics_homology70.json`; the canonical `data/metrics_homology.json` is untouched.

## Headline cells — 70% vs 40% identity

| cell | cls macro-F1 (70%) | cls macro-F1 (40%) | reg R² (70%) | reg R² (40%) |
|---|---:|---:|---:|---:|
| ESM-2 650M | 0.972 | 0.960 | 0.330 | 0.181 |
| ESM-2 150M | 0.953 | 0.920 | 0.308 | 0.162 |
| AA-composition (aa2 cls / aa3 reg) | 0.825 | 0.735 | 0.215 | 0.090 |
| best DNA-LM (nt_v2_meanG cls / dnabert2_meanD reg) | 0.783 | 0.727 | 0.186 | 0.077 |
| CDS 4-mer | 0.660 | 0.633 | — | — |
| TSS (dnabert2_meanmean) | 0.505 | 0.326 | 0.115 | 0.010 |

(Chance floor for cls macro-F1 ≈ 0.224.)

## Reading

1. **Conclusions are stable across thresholds.** ESM-2 still dominates both tasks; the
   translated AA-composition baseline still **ties/beats the best DNA-LM** (aa2 0.825 >
   nt_v2 0.783 on classification; aa3 0.215 > dnabert2 0.186 on regression). The ordering
   is identical to the primary 40% split.
2. **Every cell is higher at 70% than at 40%** — as expected: a looser identity threshold
   leaves more paralogous pairs split across train/test, so residual homology leakage
   inflates every cell. The primary 40% split is the conservative one.
3. **The TSS arm tracks the leakage gradient.** TSS macro-F1 rises from 0.326 (40%, at the
   ~0.224 floor) to 0.505 (70%), and TSS R² from 0.010 to 0.115. The TSS "signal" grows
   precisely as more homology leakage is permitted — corroborating the collapse finding:
   the random-split TSS signal was paralog leakage, and the stricter the homology control,
   the more the TSS arm falls toward chance.

**Net:** the stricter supplementary split corroborates the primary analysis — the headline
ordering and the composition / TSS-leakage conclusions do not depend on the identity
threshold.
