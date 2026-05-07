# Main Family5

Best cached family5 cell per encoder plus baselines.

| encoder | pooling | feature_source | test_macro_f1 | delta_f1_vs_4mer | test_kappa | delta_kappa_vs_4mer | test_accuracy | delta_accuracy_vs_4mer | C | timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmer | baseline | kmer | 0.6722 | 0.0000 | 0.7024 | 0.0000 | 0.8172 | 0.0000 | 1000.0000 | 2026-04-29T22:22:42+00:00 |
| dnabert2 | meanD | dnabert2_meanD | 0.7380 | 0.0657 | 0.7226 | 0.0202 | 0.8234 | 0.0062 | 10.0000 | 2026-04-30T07:26:56+00:00 |
| nt_v2 | meanD | nt_v2_meanD | 0.8275 | 0.1553 | 0.8214 | 0.1190 | 0.8871 | 0.0698 | 1.0000 | 2026-04-30T07:27:31+00:00 |
| gena_lm | clsmean | gena_lm_clsmean | 0.4982 | -0.1740 | 0.4776 | -0.2248 | 0.6858 | -0.1314 | 1.0000 | 2026-05-05T19:40:53+00:00 |
| hyena_dna | meanG | hyena_dna_meanG | 0.7149 | 0.0426 | 0.6944 | -0.0080 | 0.8090 | -0.0082 | 10.0000 | 2026-05-05T20:16:09+00:00 |
| shuffled | anti_baseline | shuffled | 0.2078 | -0.4644 | 0.0482 | -0.6542 | 0.4353 | -0.3819 | 100.0000 | 2026-04-29T22:23:07+00:00 |
