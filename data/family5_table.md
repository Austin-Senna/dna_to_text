# Family5 Model Expansion Table

Main DNA encoder comparison. Binary tasks are legacy/appendix only.

| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |
|---|---:|---:|---:|---:|
| `kmer` | 0.6722 | - | 0.8172 | 0.1743 |
| `dnabert2_meanmean` | 0.7220 | 0.7091 | 0.8172 | 0.2029 |
| `dnabert2_meanD` | 0.7380 | 0.7226 | 0.8234 | 0.2100 |
| `dnabert2_meanG` | 0.7275 | 0.7066 | 0.8131 | 0.2104 |
| `nt_v2_meanmean` | 0.7997 | 0.7982 | 0.8727 | 0.1932 |
| `nt_v2_meanD` | 0.8275 | 0.8214 | 0.8871 | 0.1882 |
| `nt_v2_meanG` | 0.8257 | 0.8179 | 0.8850 | 0.1902 |
| `gena_lm` | 0.4940 | 0.4568 | 0.6550 | 0.1173 |
| `gena_lm_meanmean` | 0.4940 | 0.4568 | 0.6550 | 0.1173 |
| `gena_lm_meanD` | 0.4843 | 0.4668 | 0.6653 | 0.1084 |
| `gena_lm_meanG` | 0.4853 | 0.4899 | 0.6961 | 0.1118 |
| `caduceus_ps` | - | - | - | - |
| `caduceus_ps_meanmean` | - | - | - | - |
| `caduceus_ps_meanD` | - | - | - | - |
| `caduceus_ps_meanG` | - | - | - | - |

## Enformer Supervised Sequence-To-Function Comparator

| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |
|---|---:|---:|---:|---:|
| `enformer_tss_4mer` | - | - | - | - |
| `enformer_trunk_global` | - | - | - | - |
| `enformer_trunk_center` | - | - | - | - |
| `enformer_tracks_center` | - | - | - | - |
