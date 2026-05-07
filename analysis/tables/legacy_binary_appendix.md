# Legacy Binary Appendix

Cached binary-task results retained for appendix use.

| encoder | pooling | feature_source | test_macro_f1 | test_kappa | test_accuracy | C | timestamp | task | shuffled_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmer | baseline | kmer | 0.9607 | 0.9213 | 0.9607 | 1000.0000 | 2026-04-29T22:22:43+00:00 | tf_vs_gpcr | False |
| dnabert2 | base | dnabert2 | 0.9381 | 0.8764 | 0.9382 | 1.0000 | 2026-04-29T22:22:35+00:00 | tf_vs_gpcr | False |
| dnabert2 | meanmean | dnabert2_meanmean | 0.9888 | 0.9775 | 0.9888 | 10.0000 | 2026-04-30T07:26:40+00:00 | tf_vs_gpcr | False |
| dnabert2 | meanD | dnabert2_meanD | 0.9719 |  | 0.9719 | 1.0000 | 2026-04-30T07:26:57+00:00 | tf_vs_gpcr | False |
| dnabert2 | meanG | dnabert2_meanG | 0.9606 |  | 0.9607 | 1000.0000 | 2026-04-30T07:27:08+00:00 | tf_vs_gpcr | False |
| dnabert2 | maxmean | dnabert2_maxmean | 0.9438 |  | 0.9438 | 10.0000 | 2026-04-30T07:26:46+00:00 | tf_vs_gpcr | False |
| dnabert2 | clsmean | dnabert2_clsmean | 0.9719 |  | 0.9719 | 10.0000 | 2026-04-30T07:26:50+00:00 | tf_vs_gpcr | False |
| nt_v2 | base | nt_v2 | 0.9775 | 0.9551 | 0.9775 | 0.1000 | 2026-04-29T22:22:39+00:00 | tf_vs_gpcr | False |
| nt_v2 | meanmean | nt_v2_meanmean | 0.9831 |  | 0.9831 | 10.0000 | 2026-04-30T07:27:12+00:00 | tf_vs_gpcr | False |
| nt_v2 | meanD | nt_v2_meanD | 0.9888 | 0.9775 | 0.9888 | 10.0000 | 2026-04-30T07:27:32+00:00 | tf_vs_gpcr | False |
| nt_v2 | meanG | nt_v2_meanG | 0.9663 |  | 0.9663 | 0.0100 | 2026-04-30T07:27:46+00:00 | tf_vs_gpcr | False |
| nt_v2 | maxmean | nt_v2_maxmean | 0.9719 |  | 0.9719 | 0.1000 | 2026-04-30T07:27:17+00:00 | tf_vs_gpcr | False |
| nt_v2 | clsmean | nt_v2_clsmean | 0.9382 |  | 0.9382 | 100.0000 | 2026-04-30T07:27:22+00:00 | tf_vs_gpcr | False |
| shuffled | baseline | shuffled | 0.4659 | -0.0674 | 0.4663 | 0.1000 | 2026-04-29T22:23:08+00:00 | tf_vs_gpcr | True |
| length | baseline | length | 0.7341 |  | 0.7360 | 1.0000 | 2026-04-29T22:22:45+00:00 | tf_vs_gpcr | False |
| kmer | baseline | kmer | 0.8392 | 0.6786 | 0.8393 | 1000.0000 | 2026-04-29T22:22:43+00:00 | tf_vs_kinase | False |
| dnabert2 | base | dnabert2 | 0.8330 |  | 0.8333 | 1.0000 | 2026-04-29T22:22:36+00:00 | tf_vs_kinase | False |
| dnabert2 | meanmean | dnabert2_meanmean | 0.8869 | 0.7738 | 0.8869 | 10.0000 | 2026-04-30T07:26:41+00:00 | tf_vs_kinase | False |
| dnabert2 | meanD | dnabert2_meanD | 0.8869 | 0.7738 | 0.8869 | 10.0000 | 2026-04-30T07:26:58+00:00 | tf_vs_kinase | False |
| dnabert2 | meanG | dnabert2_meanG | 0.8750 |  | 0.8750 | 10.0000 | 2026-04-30T07:27:09+00:00 | tf_vs_kinase | False |
| dnabert2 | maxmean | dnabert2_maxmean | 0.7911 |  | 0.7917 | 1.0000 | 2026-04-30T07:26:47+00:00 | tf_vs_kinase | False |
| dnabert2 | clsmean | dnabert2_clsmean | 0.8333 |  | 0.8333 | 100.0000 | 2026-04-30T07:26:51+00:00 | tf_vs_kinase | False |
| nt_v2 | base | nt_v2 | 0.8444 | 0.6905 | 0.8452 | 100.0000 | 2026-04-29T22:22:40+00:00 | tf_vs_kinase | False |
| nt_v2 | meanmean | nt_v2_meanmean | 0.8447 |  | 0.8452 | 100.0000 | 2026-04-30T07:27:13+00:00 | tf_vs_kinase | False |
| nt_v2 | meanD | nt_v2_meanD | 0.8444 |  | 0.8452 | 1000.0000 | 2026-04-30T07:27:33+00:00 | tf_vs_kinase | False |
| nt_v2 | meanG | nt_v2_meanG | 0.8508 |  | 0.8512 | 100.0000 | 2026-04-30T07:27:47+00:00 | tf_vs_kinase | False |
| nt_v2 | maxmean | nt_v2_maxmean | 0.8333 |  | 0.8333 | 0.1000 | 2026-04-30T07:27:18+00:00 | tf_vs_kinase | False |
| nt_v2 | clsmean | nt_v2_clsmean | 0.7678 |  | 0.7679 | 0.1000 | 2026-04-30T07:27:23+00:00 | tf_vs_kinase | False |
| shuffled | baseline | shuffled | 0.5474 | 0.0952 | 0.5476 | 0.1000 | 2026-04-29T22:23:09+00:00 | tf_vs_kinase | True |
| length | baseline | length | 0.6427 |  | 0.6429 | 0.0100 | 2026-04-29T22:22:46+00:00 | tf_vs_kinase | False |
