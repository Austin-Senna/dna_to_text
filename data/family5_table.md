# Family5 Model Expansion Table

Main DNA encoder comparison. Binary tasks are legacy/appendix only.

| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |
|---|---:|---:|---:|---:|
| `kmer` | 0.6722 | - | 0.8172 | 0.1743 |
| `dnabert2_meanmean` | 0.7220 | 0.7091 | 0.8172 | 0.2029 |
| `dnabert2_specialmean` | 0.7205 | 0.7051 | 0.8152 | 0.2029 |
| `dnabert2_meanD` | 0.7380 | 0.7226 | 0.8234 | 0.2100 |
| `dnabert2_meanG` | 0.7275 | 0.7066 | 0.8131 | 0.2104 |
| `nt_v2_meanmean` | 0.7997 | 0.7982 | 0.8727 | 0.1932 |
| `nt_v2_specialmean` | 0.7977 | 0.7985 | 0.8727 | 0.1931 |
| `nt_v2_meanD` | 0.8275 | 0.8214 | 0.8871 | 0.1882 |
| `nt_v2_meanG` | 0.8257 | 0.8179 | 0.8850 | 0.1902 |
| `gena_lm` | 0.4940 | 0.4568 | 0.6550 | 0.1173 |
| `gena_lm_meanmean` | 0.4940 | 0.4568 | 0.6550 | 0.1173 |
| `gena_lm_specialmean` | 0.4829 | 0.4589 | 0.6674 | 0.1124 |
| `gena_lm_meanD` | 0.4843 | 0.4668 | 0.6653 | 0.1084 |
| `gena_lm_meanG` | 0.4853 | 0.4899 | 0.6961 | 0.1118 |
| `hyena_dna` | 0.7103 | 0.6883 | 0.8049 | 0.1822 |
| `hyena_dna_meanmean` | 0.7103 | 0.6883 | 0.8049 | 0.1822 |
| `hyena_dna_specialmean` | 0.7095 | 0.6925 | 0.8070 | 0.1822 |
| `hyena_dna_meanD` | 0.6988 | 0.6830 | 0.8029 | 0.1818 |
| `hyena_dna_meanG` | 0.7149 | 0.6944 | 0.8090 | 0.1792 |

## TSS Self-Supervised Encoder Ablation

| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |
|---|---:|---:|---:|---:|
| `tss_nt_v2` | - | - | - | - |
| `tss_nt_v2_meanmean` | 0.4468 | 0.3754 | 0.6407 | 0.1174 |
| `tss_nt_v2_specialmean` | - | - | - | - |
| `tss_nt_v2_meanD` | 0.3481 | 0.2629 | 0.5339 | 0.0545 |
| `tss_nt_v2_meanG` | 0.4127 | 0.2848 | 0.5524 | 0.0605 |

## Enformer Supervised Sequence-To-Function Comparator

| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |
|---|---:|---:|---:|---:|
| `enformer_tss_4mer` | 0.2452 | 0.2050 | 0.5873 | 0.0413 |
| `enformer_trunk_global` | 0.5450 | 0.4392 | 0.6448 | 0.1389 |
| `enformer_trunk_center` | 0.5127 | 0.4541 | 0.6530 | 0.1425 |
| `enformer_tracks_center` | 0.4862 | 0.4264 | 0.6386 | 0.0135 |
