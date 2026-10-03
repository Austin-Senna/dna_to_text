# Licences

Three licences apply, by path.

| What | Licence |
|---|---|
| Code: `src/`, `scripts/`, `tests/` and every other source file | MIT (`LICENSE`) |
| Data and results: `data/`, `analysis/`, `outputs/`, figures and tables | CC BY 4.0 |
| Anything derived from NT-v2 outputs (below) | CC BY-NC-SA 4.0 |

NT-v2 (InstaDeepAI/nucleotide-transformer-v2-100m-multi-species) is released under CC BY-NC-SA 4.0, and we
read that licence as extending to features computed from its outputs. So these files carry it too:

- every file under `data/` or `outputs/` whose path contains `nt_v2`, in the file name or a directory name.
  That covers the pooled parquets and probe files git tracks (`data/dataset_nt_v2_*.parquet`,
  `data/dataset_tss_nt_v2*.parquet`, `data/probe_*nt_v2*.npz`), the unpacked caches
  (`data/chunk_reductions_v2_nt_v2/`, `data/tss_chunk_reductions_v2_nt_v2/`) and the stored predictions of
  the NT-v2 probe cells (`outputs/predictions/v2/*/nt_v2_*.npz`, `.../tss_nt_v2_*.npz`);
- in the Zenodo deposit, the tarballs whose names contain `nt_v2`.

One exception: the `chunk4mergc` and `chunk6mer` composition features
(`data/dataset_tss_nt_v2_chunk*.parquet` and their predictions) and
`analysis/tss_overlap/center_chunk_offsets_nt_v2.csv` are CC BY 4.0. They count bases in the windows, and
use only NT-v2's tokenizer to find chunk boundaries, not the model's outputs.

The other encoders' licences allow redistribution under CC BY 4.0: DNABERT-2 (Apache 2.0), GENA-LM (MIT),
HyenaDNA (BSD 3-Clause), Enformer (CC BY 4.0), ESM-2 (MIT). The GenePT embeddings are CC BY 4.0 (Zenodo
record 10833191) and the HGNC data CC0.
