# Cohen's kappa for headline classification cells

Cohen's kappa = (p_observed − p_expected) / (1 − p_expected). Chance-corrected: 0 = chance, 1 = perfect. Computed from the saved 5-way confusion matrices (`data/confusion_5way_*.json`); for the binary tasks we refit the probe at the recorded best C and compute kappa via `sklearn.metrics.cohen_kappa_score`.

## family5

| Dataset | C | macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| `nt_v2_meanD` | 1 | 0.8275 | **0.8214** |
| `nt_v2` | 1 | 0.8031 | **0.7984** |
| `nt_v2_meanmean` | 1 | 0.7997 | **0.7982** |
| `dnabert2_meanD` | 10 | 0.7380 | **0.7226** |
| `dnabert2_meanmean` | 10 | 0.7220 | **0.7091** |
| `kmer` | 1000 | 0.6722 | **0.7024** |
| `dnabert2` | 10 | 0.6490 | **0.6356** |
| `nt_v2_maxmean` | 0.1 | 0.5865 | **0.6099** |
| `nt_v2_clsmean` | 1 | 0.5927 | **0.5689** |
| `shuffled-label` (anti-baseline) | 100 | 0.2078 | +0.0482 |

## tf_vs_gpcr

| Dataset | C | macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| `dnabert2_meanmean` | 10 | 0.9888 | **0.9775** |
| `nt_v2_meanD` | 10 | 0.9888 | **0.9775** |
| `nt_v2` | 0.1 | 0.9775 | **0.9551** |
| `kmer` | 1000 | 0.9607 | **0.9213** |
| `dnabert2` | 1 | 0.9381 | **0.8764** |
| `shuffled-label` (anti-baseline) | 0.1 | 0.4659 | -0.0674 |

## tf_vs_kinase

| Dataset | C | macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| `dnabert2_meanmean` | 10 | 0.8869 | **0.7738** |
| `dnabert2_meanD` | 10 | 0.8869 | **0.7738** |
| `nt_v2` | 100 | 0.8444 | **0.6905** |
| `kmer` | 1000 | 0.8392 | **0.6786** |
| `shuffled-label` (anti-baseline) | 0.1 | 0.5474 | +0.0952 |

