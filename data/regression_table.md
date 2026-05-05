# Ridge-To-GenePT Regression Table

Secondary cross-modal probe: frozen sequence features are mapped to 1536-d GenePT summary embeddings with Ridge regression. `Delta vs 4-mer` is computed against the CDS 4-mer baseline.

| Feature source | Ridge R2 | Delta vs 4-mer | Mean cosine | Median cosine | Alpha |
|---|---:|---:|---:|---:|---:|
| `kmer` | 0.1743 | +0.0000 | 0.9306 | 0.9223 | 0.010 |
| `dnabert2` | 0.1812 | +0.0070 | 0.9313 | 0.9221 | 10.000 |
| `dnabert2_meanmean` | 0.2029 | +0.0286 | 0.9333 | 0.9245 | 10.000 |
| `dnabert2_meanD` | 0.2100 | +0.0357 | 0.9340 | 0.9257 | 10.000 |
| `dnabert2_meanG` | 0.2104 | +0.0361 | 0.9340 | 0.9259 | 10.000 |
| `dnabert2_maxmean` | 0.1606 | -0.0136 | 0.9293 | 0.9207 | 100.000 |
| `dnabert2_clsmean` | 0.1911 | +0.0168 | 0.9322 | 0.9235 | 100.000 |
| `nt_v2` | 0.1926 | +0.0183 | 0.9324 | 0.9254 | 10.000 |
| `nt_v2_meanmean` | 0.1932 | +0.0189 | 0.9324 | 0.9255 | 10.000 |
| `nt_v2_meanD` | 0.1882 | +0.0139 | 0.9319 | 0.9252 | 100.000 |
| `nt_v2_meanG` | 0.1902 | +0.0159 | 0.9321 | 0.9251 | 100.000 |
| `nt_v2_maxmean` | 0.1355 | -0.0387 | 0.9271 | 0.9205 | 100.000 |
| `nt_v2_clsmean` | 0.1172 | -0.0571 | 0.9251 | 0.9191 | 100.000 |
| `gena_lm` | 0.1173 | -0.0569 | 0.9251 | 0.9185 | 100.000 |
| `gena_lm_meanmean` | 0.1173 | -0.0569 | 0.9251 | 0.9185 | 100.000 |
| `gena_lm_meanD` | 0.1084 | -0.0658 | 0.9244 | 0.9173 | 100.000 |
| `gena_lm_meanG` | 0.1118 | -0.0624 | 0.9245 | 0.9185 | 1000.000 |
| `gena_lm_maxmean` | 0.0400 | -0.1342 | 0.9178 | 0.9135 | 1000.000 |
| `gena_lm_clsmean` | 0.1093 | -0.0650 | 0.9242 | 0.9185 | 100.000 |
| `caduceus_ps` | - | - | - | - | - |
| `caduceus_ps_meanmean` | - | - | - | - | - |
| `caduceus_ps_meanD` | - | - | - | - | - |
| `caduceus_ps_meanG` | - | - | - | - | - |

## Enformer Supervised Sequence-To-Function Comparator

| Feature source | Ridge R2 | Delta vs 4-mer | Mean cosine | Median cosine | Alpha |
|---|---:|---:|---:|---:|---:|
| `enformer_tss_4mer` | - | - | - | - | - |
| `enformer_trunk_global` | - | - | - | - | - |
| `enformer_trunk_center` | - | - | - | - | - |
| `enformer_tracks_center` | - | - | - | - | - |

## Best Observed Regression Cell

| Feature source | Ridge R2 | Mean cosine |
|---|---:|---:|
| `dnabert2_meanG` | 0.2104 | 0.9340 |
