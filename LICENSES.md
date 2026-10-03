# Licences

Three licences apply, by path.

| What | Licence |
|---|---|
| Code: `src/`, `scripts/`, `tests/` and every other source file | MIT (`LICENSE`) |
| Data and results: `data/`, `analysis/`, `outputs/`, figures and tables | CC BY 4.0 |
| Anything derived from NT-v2 outputs (below) | CC BY-NC-SA 4.0 |

NT-v2 (InstaDeepAI/nucleotide-transformer-v2-100m-multi-species) is released under CC BY-NC-SA 4.0, and we
read that licence as extending to features computed from its outputs. So these files carry it too:

- in this repository, every file under `data/` whose name contains `nt_v2`
  (`dataset_nt_v2_*.parquet`, `dataset_tss_nt_v2*.parquet`, `probe_nt_v2*.npz`, `probe_tss_nt_v2*.npz`),
  except the `chunk4mergc` and `chunk6mer` composition features, which are computed from DNA alone;
- in the Zenodo deposit, the tarballs whose names contain `nt_v2`.

The other encoders' licences allow redistribution under CC BY 4.0: DNABERT-2 (Apache 2.0), GENA-LM (MIT),
HyenaDNA (BSD 3-Clause), Enformer (CC BY 4.0), ESM-2 (MIT). The GenePT embeddings are CC BY 4.0 (Zenodo
record 10833191) and the HGNC data CC0.
