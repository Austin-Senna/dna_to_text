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

## Δ vs 4-mer baseline

Same matrix, but each cell is `(value) − (kmer value)` in the same column. Positive = beats k-mer composition; negative = worse than k-mer; zero = tied. k-mer's own row reads zero by construction. Reading this table answers *"how much extra signal does the encoder + pooling carry over a 256-d 4-mer histogram?"*

| Feature source | Δ 5-way F1 | Δ 5-way κ | Δ tf-vs-gpcr F1 | Δ tf-vs-gpcr κ | Δ tf-vs-kinase F1 | Δ tf-vs-kinase κ | Δ Ridge R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shuffled` | -0.4644 | -0.6541 | -0.4948 | -0.9888 | -0.2919 | -0.5833 | — |
| `length` | -0.5340 | -0.7130 | -0.2266 | -0.4494 | -0.1966 | -0.3929 | — |
| `kmer` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `dnabert2` | -0.0232 | -0.0667 | -0.0226 | -0.0449 | -0.0063 | -0.0119 | +0.0070 |
| `dnabert2_meanmean` | +0.0497 | +0.0067 | +0.0281 | +0.0562 | +0.0477 | +0.0952 | +0.0286 |
| `dnabert2_meanD` | +0.0657 | +0.0202 | +0.0112 | +0.0225 | +0.0477 | +0.0952 | +0.0357 |
| `dnabert2_meanG` | +0.0552 | +0.0042 | -0.0000 | +0.0000 | +0.0358 | +0.0714 | +0.0361 |
| `dnabert2_maxmean` | -0.0578 | -0.1207 | -0.0169 | -0.0337 | -0.0482 | -0.0952 | -0.0136 |
| `dnabert2_clsmean` | -0.0175 | -0.0407 | +0.0112 | +0.0225 | -0.0059 | -0.0119 | +0.0168 |
| `nt_v2` | +0.1308 | +0.0960 | +0.0169 | +0.0337 | +0.0052 | +0.0119 | +0.0183 |
| `nt_v2_meanmean` | +0.1274 | +0.0959 | +0.0225 | +0.0449 | +0.0055 | +0.0119 | +0.0189 |
| `nt_v2_meanD` | +0.1553 | +0.1190 | +0.0281 | +0.0562 | +0.0052 | +0.0119 | +0.0139 |
| `nt_v2_meanG` | +0.1535 | +0.1155 | +0.0056 | +0.0112 | +0.0115 | +0.0238 | +0.0159 |
| `nt_v2_maxmean` | -0.0857 | -0.0925 | +0.0112 | +0.0225 | -0.0059 | -0.0119 | -0.0387 |
| `nt_v2_clsmean` | -0.0795 | -0.1335 | -0.0225 | -0.0449 | -0.0714 | -0.1429 | -0.0571 |

**Cells beating 4-mer by Δ macro-F1 ≥ 0.02** (the spec's decision-gate threshold from `2026-04-29-classification-pivot-design.md`):

| Feature source | 5-way | tf-vs-gpcr | tf-vs-kinase |
|---|:-:|:-:|:-:|
| `length` | ❌ | ❌ | ❌ |
| `dnabert2` | ❌ | ❌ | ≈ |
| `dnabert2_meanmean` | ✅ | ✅ | ✅ |
| `dnabert2_meanD` | ✅ | ≈ | ✅ |
| `dnabert2_meanG` | ✅ | ≈ | ✅ |
| `dnabert2_maxmean` | ❌ | ≈ | ❌ |
| `dnabert2_clsmean` | ≈ | ≈ | ≈ |
| `nt_v2` | ✅ | ≈ | ≈ |
| `nt_v2_meanmean` | ✅ | ✅ | ≈ |
| `nt_v2_meanD` | ✅ | ✅ | ≈ |
| `nt_v2_meanG` | ✅ | ≈ | ≈ |
| `nt_v2_maxmean` | ❌ | ≈ | ≈ |
| `nt_v2_clsmean` | ❌ | ❌ | ❌ |

Legend: ✅ beats k-mer (Δ ≥ +0.02); ≈ ties (within ±0.02); ❌ loses (Δ ≤ −0.02).

