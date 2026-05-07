# Main Regression

Best cached Ridge-to-GenePT cell per encoder plus 4-mer baseline.

| encoder | pooling | feature_source | test_r2_macro | delta_vs_4mer | test_mean_cosine | test_median_cosine | alpha | timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmer | baseline | kmer | 0.1743 | 0.0000 | 0.9306 | 0.9223 | 0.0100 | 2026-04-18T20:25:47+00:00 |
| dnabert2 | meanG | dnabert2_meanG | 0.2104 | 0.0361 | 0.9340 | 0.9259 | 10.0000 | 2026-04-30T07:38:33+00:00 |
| nt_v2 | meanmean | nt_v2_meanmean | 0.1932 | 0.0189 | 0.9324 | 0.9255 | 10.0000 | 2026-04-30T07:38:34+00:00 |
| gena_lm | base | gena_lm | 0.1173 | -0.0569 | 0.9251 | 0.9185 | 100.0000 | 2026-05-05T19:41:12+00:00 |
| hyena_dna | base | hyena_dna | 0.1822 | 0.0079 | 0.9313 | 0.9236 | 1.0000 | 2026-05-05T20:17:49+00:00 |
