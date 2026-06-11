# α-selection sensitivity (MINA #3)

_Source: `metrics_homology.json` · homology (40%) split · 53 regression cells._

Does the GenePT-regression ranking depend on whether Ridge α is selected by validation **mean-cosine** or validation **macro-R²**? Cells where the two rules pick the same α have identical test R² by construction; cells where α differs were re-run with the canonical protocol (`--select-by cosine`, refit on train+val, evaluate on test).

**50/53 cells select the same α under both rules.** 3 cell(s) differ; **2 cell(s) change rank** — the leaders are unchanged.

| rank (R²) | cell | α (R²-sel) | test R² (R²-sel) | α (cos-sel) | test R² (cos-sel) | rank (cos) | α differs |
|---:|---|---:|---:|---:|---:|---:|:--:|
| 1 | `esm2_650m` | 10 | +0.1813 | 10 | +0.1813 | 1 |  |
| 2 | `esm2_150m` | 10 | +0.1622 | 10 | +0.1622 | 2 |  |
| 3 | `aa3` | 0.01 | +0.0904 | 0.01 | +0.0904 | 3 |  |
| 4 | `dnabert2_meanD` | 10 | +0.0766 | 10 | +0.0766 | 4 |  |
| 5 | `dnabert2_meanG` | 10 | +0.0756 | 10 | +0.0756 | 5 |  |
| 6 | `dnabert2_specialmean` | 10 | +0.0733 | 10 | +0.0733 | 6 |  |
| 7 | `dnabert2_meanmean` | 10 | +0.0733 | 10 | +0.0733 | 7 |  |
| 8 | `dnabert2_clsmean` | 100 | +0.0617 | 100 | +0.0617 | 8 |  |
| 9 | `aa2` | 0.1 | +0.0604 | 0.1 | +0.0604 | 9 |  |
| 10 | `kmer6` | 0.01 | +0.0570 | 0.01 | +0.0570 | 10 |  |
| 11 | `nt_v2_meanG` | 100 | +0.0522 | 100 | +0.0522 | 11 |  |
| 12 | `nt_v2_meanD` | 100 | +0.0513 | 100 | +0.0513 | 12 |  |
| 13 | `nt_v2_meanmean` | 10 | +0.0512 | 10 | +0.0512 | 13 |  |
| 14 | `nt_v2_specialmean` | 10 | +0.0511 | 10 | +0.0511 | 14 |  |
| 15 | `kmer` | 0.01 | +0.0448 | 0.01 | +0.0448 | 15 |  |
| 16 | `codon` | 0.1 | +0.0441 | 0.1 | +0.0441 | 16 |  |
| 17 | `dnabert2_maxmean` | 100 | +0.0437 | 100 | +0.0437 | 17 |  |
| 18 | `hyena_dna_meanG` | 10 | +0.0402 | 10 | +0.0402 | 18 |  |
| 19 | `hyena_dna_meanD` | 10 | +0.0399 | 10 | +0.0399 | 19 |  |
| 20 | `hyena_dna_specialmean` | 1 | +0.0385 | 1 | +0.0385 | 20 |  |
| 21 | `hyena_dna_meanmean` | 1 | +0.0385 | 1 | +0.0385 | 21 |  |
| 22 | `aa1` | 1 | +0.0308 | 1 | +0.0308 | 22 |  |
| 23 | `nt_v2_maxmean` | 1000 | +0.0266 | 1000 | +0.0266 | 23 |  |
| 24 | `tss_dnabert2_meanmean` | 0.1 | +0.0102 | 0.1 | +0.0102 | 24 |  |
| 25 | `hyena_dna_maxmean` | 100 | +0.0094 | 100 | +0.0094 | 25 |  |
| 26 | `nt_v2_clsmean` | 1000 | +0.0086 | 1000 | +0.0086 | 26 |  |
| 27 | `tss_nt_v2_meanmean` | 1 | +0.0027 | 1 | +0.0027 | 27 |  |
| 28 | `gena_lm_meanG` | 1000 | +0.0022 | 1000 | +0.0022 | 28 |  |
| 29 | `gena_lm_meanD` | 1000 | +0.0016 | 1000 | +0.0016 | 29 |  |
| 30 | `gena_lm_clsmean` | 100 | +0.0003 | 100 | +0.0003 | 30 |  |
| 31 | `gena_lm_meanmean` | 100 | -0.0006 | 100 | -0.0006 | 31 |  |
| 32 | `gena_lm_specialmean` | 100 | -0.0029 | 100 | -0.0029 | 32 |  |
| 33 | `tss_dnabert2_maxmean` | 1 | -0.0051 | 1 | -0.0051 | 33 |  |
| 34 | `tss_dnabert2_clsmean` | 10 | -0.0062 | 10 | -0.0062 | 34 |  |
| 35 | `tss_dnabert2_meanG` | 10 | -0.0115 | 10 | -0.0115 | 35 |  |
| 36 | `tss_nt_v2_maxmean` | 100 | -0.0139 | 10 | -0.0143 | 36 | yes |
| 37 | `tss_hyena_dna_maxmean` | 1000 | -0.0158 | 1000 | -0.0158 | 37 |  |
| 38 | `tss_dnabert2_meanD` | 100 | -0.0176 | 100 | -0.0176 | 38 |  |
| 39 | `tss_hyena_dna_meanmean` | 0.1 | -0.0180 | 0.1 | -0.0180 | 39 |  |
| 40 | `tss_nt_v2_meanG` | 1000 | -0.0204 | 1000 | -0.0204 | 40 |  |
| 41 | `tss_nt_v2_clsmean` | 100 | -0.0205 | 100 | -0.0205 | 41 |  |
| 42 | `tss_gena_lm_meanmean` | 100 | -0.0207 | 100 | -0.0207 | 42 |  |
| 43 | `tss_gena_lm_clsmean` | 100 | -0.0212 | 100 | -0.0212 | 43 |  |
| 44 | `tss_gena_lm_maxmean` | 1000 | -0.0214 | 1000 | -0.0214 | 44 |  |
| 45 | `tss_gena_lm_meanG` | 1000 | -0.0226 | 1000 | -0.0226 | 45 |  |
| 46 | `gc` | 1000 | -0.0226 | 1000 | -0.0226 | 46 |  |
| 47 | `tss_hyena_dna_meanD` | 1000 | -0.0226 | 1000 | -0.0226 | 47 |  |
| 48 | `tss_nt_v2_meanD` | 1000 | -0.0228 | 1000 | -0.0228 | 48 |  |
| 49 | `tss_gena_lm_meanD` | 1000 | -0.0230 | 1000 | -0.0230 | 49 |  |
| 50 | `tss_hyena_dna_meanG` | 1000 | -0.0232 | 1000 | -0.0232 | 50 |  |
| 51 | `hyena_dna_clsmean` | 1 | -0.0241 | 0.01 | -0.0241 | 52 | yes |
| 52 | `tss_hyena_dna_clsmean` | 1 | -0.0241 | 0.01 | -0.0241 | 51 | yes |
| 53 | `gena_lm_maxmean` | 1000 | -0.0594 | 1000 | -0.0594 | 53 |  |
