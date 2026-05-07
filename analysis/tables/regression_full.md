# Regression Full

All cached Ridge cells with delta versus 4-mer.

| encoder | pooling | feature_source | test_r2_macro | delta_vs_4mer | test_mean_cosine | test_median_cosine | alpha | timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmer | baseline | kmer | 0.1743 | 0.0000 | 0.9306 | 0.9223 | 0.0100 | 2026-04-18T20:25:47+00:00 |
| dnabert2 | base | dnabert2 | 0.1812 | 0.0070 | 0.9313 | 0.9221 | 10.0000 | 2026-04-18T19:53:58+00:00 |
| dnabert2 | meanmean | dnabert2_meanmean | 0.2029 | 0.0286 | 0.9333 | 0.9245 | 10.0000 | 2026-04-30T07:38:29+00:00 |
| dnabert2 | meanD | dnabert2_meanD | 0.2100 | 0.0357 | 0.9340 | 0.9257 | 10.0000 | 2026-04-30T07:38:32+00:00 |
| dnabert2 | meanG | dnabert2_meanG | 0.2104 | 0.0361 | 0.9340 | 0.9259 | 10.0000 | 2026-04-30T07:38:33+00:00 |
| dnabert2 | maxmean | dnabert2_maxmean | 0.1606 | -0.0136 | 0.9293 | 0.9207 | 100.0000 | 2026-04-30T07:38:30+00:00 |
| dnabert2 | clsmean | dnabert2_clsmean | 0.1911 | 0.0168 | 0.9322 | 0.9235 | 100.0000 | 2026-04-30T07:38:30+00:00 |
| nt_v2 | base | nt_v2 | 0.1926 | 0.0183 | 0.9324 | 0.9254 | 10.0000 | 2026-04-18T22:58:59+00:00 |
| nt_v2 | meanmean | nt_v2_meanmean | 0.1932 | 0.0189 | 0.9324 | 0.9255 | 10.0000 | 2026-04-30T07:38:34+00:00 |
| nt_v2 | meanD | nt_v2_meanD | 0.1882 | 0.0139 | 0.9319 | 0.9252 | 100.0000 | 2026-04-30T07:38:37+00:00 |
| nt_v2 | meanG | nt_v2_meanG | 0.1902 | 0.0159 | 0.9321 | 0.9251 | 100.0000 | 2026-04-30T07:38:39+00:00 |
| nt_v2 | maxmean | nt_v2_maxmean | 0.1355 | -0.0387 | 0.9271 | 0.9205 | 100.0000 | 2026-04-30T07:38:35+00:00 |
| nt_v2 | clsmean | nt_v2_clsmean | 0.1172 | -0.0571 | 0.9251 | 0.9191 | 100.0000 | 2026-04-30T07:38:36+00:00 |
| gena_lm | base | gena_lm | 0.1173 | -0.0569 | 0.9251 | 0.9185 | 100.0000 | 2026-05-05T19:41:12+00:00 |
| gena_lm | meanmean | gena_lm_meanmean | 0.1173 | -0.0569 | 0.9251 | 0.9185 | 100.0000 | 2026-05-05T19:41:18+00:00 |
| gena_lm | meanD | gena_lm_meanD | 0.1084 | -0.0658 | 0.9244 | 0.9173 | 100.0000 | 2026-05-05T19:41:25+00:00 |
| gena_lm | meanG | gena_lm_meanG | 0.1118 | -0.0624 | 0.9245 | 0.9185 | 1000.0000 | 2026-05-05T19:41:38+00:00 |
| gena_lm | maxmean | gena_lm_maxmean | 0.0400 | -0.1342 | 0.9178 | 0.9135 | 1000.0000 | 2026-05-05T19:43:21+00:00 |
| gena_lm | clsmean | gena_lm_clsmean | 0.1093 | -0.0650 | 0.9242 | 0.9185 | 100.0000 | 2026-05-05T19:43:27+00:00 |
| hyena_dna | base | hyena_dna | 0.1822 | 0.0079 | 0.9313 | 0.9236 | 1.0000 | 2026-05-05T20:17:49+00:00 |
| hyena_dna | meanmean | hyena_dna_meanmean | 0.1822 | 0.0079 | 0.9313 | 0.9236 | 1.0000 | 2026-05-05T20:17:50+00:00 |
| hyena_dna | meanD | hyena_dna_meanD | 0.1818 | 0.0075 | 0.9314 | 0.9235 | 1.0000 | 2026-05-05T20:17:51+00:00 |
| hyena_dna | meanG | hyena_dna_meanG | 0.1792 | 0.0049 | 0.9310 | 0.9233 | 10.0000 | 2026-05-05T20:17:52+00:00 |
| hyena_dna | maxmean | hyena_dna_maxmean | 0.1329 | -0.0414 | 0.9266 | 0.9193 | 100.0000 | 2026-05-05T20:17:53+00:00 |
| hyena_dna | clsmean | hyena_dna_clsmean | -0.0015 | -0.1758 | 0.9129 | 0.9131 | 0.0100 | 2026-05-05T20:17:54+00:00 |
| enformer_tracks_center | baseline | enformer_tracks_center | 0.0135 | -0.1607 | 0.9171 | 0.9108 | 1000.0000 | 2026-05-06T19:58:37+00:00 |
| enformer_trunk_center | baseline | enformer_trunk_center | 0.1425 | -0.0318 | 0.9275 | 0.9200 | 1000.0000 | 2026-05-06T19:58:25+00:00 |
| enformer_trunk_global | baseline | enformer_trunk_global | 0.1389 | -0.0353 | 0.9271 | 0.9192 | 100.0000 | 2026-05-06T19:58:13+00:00 |
| enformer_tss_4mer | baseline | enformer_tss_4mer | 0.0413 | -0.1329 | 0.9173 | 0.9148 | 0.0100 | 2026-05-06T16:49:59+00:00 |
| tss_nt_v2_meanD | baseline | tss_nt_v2_meanD | 0.0545 | -0.1197 | 0.9190 | 0.9142 | 10.0000 | 2026-05-07T02:57:22+00:00 |
| tss_nt_v2_meanG | baseline | tss_nt_v2_meanG | 0.0605 | -0.1138 | 0.9197 | 0.9146 | 10.0000 | 2026-05-07T03:15:01+00:00 |
| tss_nt_v2_meanmean | baseline | tss_nt_v2_meanmean | 0.1174 | -0.0569 | 0.9252 | 0.9169 | 0.1000 | 2026-05-07T02:47:47+00:00 |
