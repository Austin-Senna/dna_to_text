# Homology-split re-run — what changed and why (for Austin)

**TL;DR:** Under the homology-aware split that *Bioinformatics* requires, two of our headline claims weaken. NT-v2 still beats the weak CDS 4-mer baseline, but it is **statistically tied with a translated amino-acid composition baseline** on classification and **loses to it** on the GenePT-regression arm. We need to reframe "DNA-LMs recover protein-family signal beyond composition."

## What was done
- New PRIMARY split: MMseqs2 protein clusters @40% id, whole clusters assigned to one split (0 clusters span train/val/test). 3,244 genes → **1,751 clusters** — i.e. ~1,500 genes had a paralog the old random split was free to leak across train/test. Random split kept as sensitivity (`splits_random.json`).
- Re-ran all 125 probe cells on the homology split (no re-encoding; embeddings are split-independent). Results in `data/metrics_homology.json`; old random-split results preserved in `data/metrics.json`.
- α now selected by validation macro-R² (was cosine); added composition baselines (6-mer, codon, translated AA 1/2/3-mer, GC+length); added paired-difference bootstrap.

## Classification — family5 macro-F1
| Cell | Homology | Random (old) |
|---|---|---|
| AA-2mer baseline | **0.735** | — |
| NT-v2 meanD (best DNA-LM) | 0.722 | 0.828 |
| AA-1mer | 0.681 | — |
| codon | 0.645 | — |
| CDS 4-mer | 0.633 | 0.672 |
| GC+length | 0.164 | — |

## Regression — R² → GenePT
| Cell | R² | cosine |
|---|---|---|
| AA-3mer baseline | **0.090** | 0.921 |
| DNABERT-2 meanD (best DNA-LM) | 0.077 | 0.920 |
| NT-v2 | 0.052 | 0.918 |
| CDS 4-mer | 0.045 | 0.917 |
| GC+length | −0.023 | 0.910 |

## Paired-difference bootstrap (1000 iters, same test genes, n=487)
| Comparison (A − B) | Δ | 95% CI | P(A>B) |
|---|---|---|---|
| cls NT-v2 − AA-2mer | −0.013 | [−0.070, +0.053] | 0.36 → **tied** |
| cls NT-v2 − 4-mer | +0.089 | [+0.025, +0.161] | 0.99 → NT-v2 wins |
| cls AA-2mer − 4-mer | +0.102 | [+0.036, +0.170] | 0.999 |
| reg DNABERT-2 − AA-3mer | −0.014 | [−0.021, −0.007] | 0.001 → **AA wins** |
| reg DNABERT-2 − 4-mer | +0.032 | [+0.026, +0.038] | 1.000 |
| reg AA-3mer − 4-mer | +0.046 | [+0.037, +0.054] | 1.000 |

Reproduce: `uv run python scripts/bootstrap_test_uncertainty.py --paired` (PAIRED_CLS/PAIRED_REG hold these comparisons).

## Three takeaways (all three MINA fixes confirmed)
1. **Homology split halves the DNA-LM-over-k-mer gain** (+0.156 → +0.089). Real leakage was inflating it.
2. **Translated AA composition matches/beats the best DNA-LM** — tied on classification, significantly better on regression. The "beyond composition" framing does not survive.
3. **Cosine is uninformative** (≈0.91–0.92 for every cell, R² −0.06…+0.09). Selecting α by R² and reporting R² is essential; cosine masked everything.

## Suggested reframing (for discussion, not yet applied to the manuscript)
- Title/abstract: shift from "DNA-LMs encode protein-family/function signal" to a more honest "DNA-LMs match but do not exceed simple compositional/translated baselines on a homology-controlled benchmark" — or pivot the contribution to the *benchmark + honest evaluation* itself.
- Keep NT-v2 > 4-mer (true, significant) but stop implying it beats *all* simple baselines.
- The ESM-2 comparator (planned) will show whether a real protein LM extends the AA-composition signal — likely yes, which strengthens a "protein-level signal, recoverable from composition" narrative.
