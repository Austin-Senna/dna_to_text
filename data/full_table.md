# Full results table — all baselines × all (encoder × pooling) variants

Every cell visited in the project, with macro-F1 + Cohen's κ for each of the three classification tasks and Ridge R² macro into GenePT 1536-d for the regression task.

Numbers traceable to `data/metrics.json` (latest entry per cell, deduped on `(encoder, task, shuffled_labels)`). κ for `family5` cells is computed from `data/confusion_5way_*.json`; for binary cells the probe is refit at the recorded best C and κ is computed via `sklearn.metrics.cohen_kappa_score`. `—` = not run for this combination.

Row groups: **baselines** (shuffled-label, length-only, 4-mer) → **Phase 1–3 originals** (no special tokens) → **Phase 4b DNABERT-2 variants** → **Phase 4b NT-v2 variants**.

| Feature source | 5-way F1 | 5-way κ | tf-vs-gpcr F1 | tf-vs-gpcr κ | tf-vs-kinase F1 | tf-vs-kinase κ | Ridge R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shuffled` | 0.2078 | +0.0482 | 0.4659 | -0.0674 | 0.5474 | +0.0952 | — |
| `length` | 0.1382 | -0.0106 | 0.7341 | 0.4719 | 0.6427 | 0.2857 | — |
| `kmer` | 0.6722 | 0.7024 | 0.9607 | 0.9213 | 0.8392 | 0.6786 | 0.1743 |
| `dnabert2` | 0.6490 | 0.6356 | 0.9381 | 0.8764 | 0.8330 | 0.6667 | 0.1812 |
| `dnabert2_meanmean` | 0.7220 | 0.7091 | 0.9888 | 0.9775 | 0.8869 | 0.7738 | 0.2029 |
| `dnabert2_meanD` | 0.7380 | 0.7226 | 0.9719 | 0.9438 | 0.8869 | 0.7738 | 0.2100 |
| `dnabert2_meanG` | 0.7275 | 0.7066 | 0.9606 | 0.9213 | 0.8750 | 0.7500 | 0.2104 |
| `dnabert2_maxmean` | 0.6144 | 0.5817 | 0.9438 | 0.8876 | 0.7911 | 0.5833 | 0.1606 |
| `dnabert2_clsmean` | 0.6547 | 0.6616 | 0.9719 | 0.9438 | 0.8333 | 0.6667 | 0.1911 |
| `nt_v2` | 0.8031 | 0.7984 | 0.9775 | 0.9551 | 0.8444 | 0.6905 | 0.1926 |
| `nt_v2_meanmean` | 0.7997 | 0.7982 | 0.9831 | 0.9663 | 0.8447 | 0.6905 | 0.1932 |
| `nt_v2_meanD` | 0.8275 | 0.8214 | 0.9888 | 0.9775 | 0.8444 | 0.6905 | 0.1882 |
| `nt_v2_meanG` | 0.8257 | 0.8179 | 0.9663 | 0.9326 | 0.8508 | 0.7024 | 0.1902 |
| `nt_v2_maxmean` | 0.5865 | 0.6099 | 0.9719 | 0.9438 | 0.8333 | 0.6667 | 0.1355 |
| `nt_v2_clsmean` | 0.5927 | 0.5689 | 0.9382 | 0.8764 | 0.7678 | 0.5357 | 0.1172 |

**Reading guide.** macro-F1 is the per-task headline metric (unweighted mean of per-class F1). Cohen's κ is chance-corrected (0 = chance, 1 = perfect; negative = worse than chance). R² macro is the unweighted mean R² across the 1536-d GenePT regression target — chance-corrected by construction (R² = 0 ≡ predict-the-mean).

**Anti-baseline interpretation.** The shuffled-label row is the empirical chance level for this exact pipeline. Across all three tasks κ falls within ±0.10 of zero — pipeline is honest.

**Best of each column.** Highlighted in bold below.

| Column | Best feature source | Value |
|---|---|---:|
| 5-way F1 | `nt_v2_meanD` | **0.8275** |
| 5-way κ | `nt_v2_meanD` | **0.8214** |
| tf-vs-gpcr F1 | `dnabert2_meanmean` | **0.9888** |
| tf-vs-gpcr κ | `dnabert2_meanmean` | **0.9775** |
| tf-vs-kinase F1 | `dnabert2_meanmean` | **0.8869** |
| tf-vs-kinase κ | `dnabert2_meanmean` | **0.7738** |
| Ridge R² | `dnabert2_meanG` | **0.2104** |

