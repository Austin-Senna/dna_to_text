# Pipeline

How `prepare_data.py` and `run_encoder.py` build `data/dataset.parquet`.

## Inputs
## Two databases, two roles

- **HGNC** = naming + family annotations (no DNA). Tells us "this gene is officially `FCGR1A`, it's an Fc receptor."
- **Ensembl** = genome coordinates + DNA sequence. Tells us the actual `ATG…` for `ENSG00000150337`.

They are joined inside the HGNC TSV: every row has both `symbol` and `ensembl_gene_id`.

GenePT is keyed by HGNC gene symbol, so the chain is:

```
GenePT symbol  ──────►  HGNC symbol  ──────►  Ensembl ID  ──────►  CDS  ──────►  DNABERT-2 vector
                       (same string)         (HGNC TSV col)        (Ensembl REST)
```

## Family selection

We don't use HGNC group IDs (they're fragmented — no master "all kinases" group). Instead we substring-match the `gene_group` text column in the HGNC TSV. Each family in `src/data_loader/dataset_loader.py::FAMILIES` is:

```python
(short_name, display_name, [include_regexes], [exclude_regexes])
```

Current families and approximate counts after the GenePT intersection:

| Family | Includes (rough) | ~count |
|---|---|---|
| `kinase` | `\bkinase` (minus inhibitors/regulators/substrates) | ~558 |
| `tf` | zinc finger, homeobox, bHLH, bZIP, forkhead, HMG, nuclear receptor, T-box, SOX, ETS, "transcription factor" | ~1745 |
| `ion` | `\bchannel` (minus regulators/auxiliary) | ~200 |
| `gpcr` | "G protein-coupled receptor", adrenoceptor, 5-HT, dopamine, muscarinic, opioid, chemokine, olfactory | ~601 |
| `immune` | TLR, IL receptor, Fc receptor, NLR, KIR, TCR, BCR, C-type lectin, Ig-like receptor | ~160 |

Cross-family duplicates are removed: first family wins (`build_gene_table` tracks `seen_ensembl`).

## `prepare_data.py` (CPU-only)

1. `analyze_genept(...)` — print pickle structure (n_genes, embedding_dim, key format)
2. `load_hgnc_complete(...)` — download HGNC TSV once, cache it, filter to `locus_group == "protein-coding gene"`
3. For each family in `FAMILIES`:
   - `filter_family(...)` — regex match on `gene_group` (include - exclude)
   - drop rows with no `ensembl_id`
   - intersect with GenePT symbol set
   - drop ensembl_ids already claimed by an earlier family
   - apply `--limit` if set
4. `fetch_all(...)` — for each Ensembl ID, GET `https://rest.ensembl.org/sequence/id/{id}?type=cds`, cache to `data/sequences/{id}.fa`, rate-limited at ~14 req/s, retries on 5xx
5. Drop any genes whose CDS fetch failed
6. Write `data/gene_table.parquet` with columns:
   `symbol, ensembl_id, family, summary, y_embedding`

## `run_encoder.py` (GPU / MPS / CPU)

1. Load `data/gene_table.parquet`
2. Re-read each cached CDS from `data/sequences/`
3. `embed_all(...)`:
   - Load DNABERT-2 + tokenizer once
   - For each CDS:
     - BPE-tokenize the full sequence (no special tokens, no truncation)
     - Window into chunks of 512 tokens with 64-token overlap
     - For each chunk: forward pass, mean of `last_hidden_state` over the token dim → 768-d
     - Mean across chunks → final 768-d gene vector
   - Cache each vector to `data/embeddings/{ENSG…}.npy`, skip already-cached
4. Attach `x` column to the DataFrame, rename `y_embedding` → `y`
5. Write `data/dataset.parquet` with columns:
   `symbol, ensembl_id, family, summary, x (768-d), y (1536-d)`

## Caching

Everything intermediate is cached so reruns are cheap:

| Cache | What | Cleared when |
|---|---|---|
| `data/hgnc/hgnc_complete_set.tsv` | full HGNC TSV | manually delete to refresh |
| `data/sequences/{ENSG}.fa` | one FASTA per gene | manually |
| `data/embeddings/{ENSG}.npy` | one 768-d vector per gene | manually |
| `data/gene_table.parquet` | post-prepare table | rebuilt by `prepare_data.py` |
| `data/dataset.parquet` | final table | rebuilt by `run_encoder.py` |

## Run order

```bash
# 1. inspect GenePT structure
python scripts/prepare_data.py --analyze-only

# 2. preview families (no data fetched)
python scripts/inspect_families.py

# 3. tiny smoke test end-to-end
python scripts/prepare_data.py --limit 5
python scripts/run_encoder.py --device mps    # or cuda / cpu
python scripts/inspect_data.py

# 4. full run
python scripts/prepare_data.py
python scripts/run_encoder.py --device mps
```
