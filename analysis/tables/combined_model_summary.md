# Combined Model Summary

Family5 and Ridge headline metrics side by side.

| encoder | family5_feature_source | family5_macro_f1 | family5_kappa | family5_accuracy | regression_feature_source | ridge_r2_macro | ridge_delta_vs_4mer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kmer | kmer | 0.6722 | 0.7024 | 0.8172 | kmer | 0.1743 | 0.0000 |
| shuffled_y |  |  |  |  | shuffled_y | -0.0004 | -0.1746 |
| dnabert2 | dnabert2_meanD | 0.7380 | 0.7226 | 0.8234 | dnabert2_meanG | 0.2104 | 0.0361 |
| nt_v2 | nt_v2_meanD | 0.8275 | 0.8214 | 0.8871 | nt_v2_meanmean | 0.1932 | 0.0189 |
| gena_lm | gena_lm_clsmean | 0.4982 | 0.4776 | 0.6858 | gena_lm | 0.1173 | -0.0569 |
| hyena_dna | hyena_dna_meanG | 0.7149 | 0.6944 | 0.8090 | hyena_dna_specialmean | 0.1822 | 0.0080 |
| shuffled | shuffled | 0.2078 | 0.0482 | 0.4353 |  |  |  |
