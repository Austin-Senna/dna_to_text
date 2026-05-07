# Context Ablation

CDS, TSS, and Enformer context comparison for family5 and Ridge probes.

| label | context | model_group | family5_feature_source | regression_feature_source | family5_macro_f1 | family5_kappa | family5_accuracy | ridge_r2_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CDS 4-mer | CDS | composition | kmer | kmer | 0.6722 | 0.7024 | 0.8172 | 0.1743 |
| CDS NT-v2 meanD | CDS | self-supervised encoder | nt_v2_meanD | nt_v2_meanD | 0.8275 | 0.8214 | 0.8871 | 0.1882 |
| TSS 4-mer | TSS window | composition | enformer_tss_4mer | enformer_tss_4mer | 0.2452 | 0.2050 | 0.5873 | 0.0413 |
| TSS NT-v2 meanmean | TSS window | self-supervised encoder | tss_nt_v2_meanmean | tss_nt_v2_meanmean | 0.4468 | 0.3754 | 0.6407 | 0.1174 |
| Enformer trunk | TSS window | supervised comparator | enformer_trunk_global | enformer_trunk_center | 0.5450 | 0.4392 | 0.6448 | 0.1425 |
