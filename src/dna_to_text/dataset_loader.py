"""Load GenePT artifacts, HGNC family lists, and join them into a gene table."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# (hgnc_group_id, short_name, display_name)
# Group IDs are best-guess starting points; verify against
# https://www.genenames.org/data/genegroup/ on first run.
FAMILIES: list[tuple[int, str, str]] = [
    (694, "kinase", "Protein kinases"),
    (1722, "tf", "Transcription factors"),
    (177, "ion", "Ion channels"),
    (139, "gpcr", "GPCRs"),
    (590, "immune", "Immune receptors"),
]

HGNC_URL = "https://www.genenames.org/cgi-bin/genegroup/download?id={group_id}&type=branch"


def analyze_genept(pickle_path: str | Path) -> dict:
    """Print and return basic structure info for a GenePT embedding pickle."""
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict-like pickle, got {type(obj).__name__}")

    keys = list(obj.keys())
    sample_keys = keys[:10]
    first_val = obj[keys[0]]
    arr = np.asarray(first_val)
    sample = sample_keys[0] if sample_keys else ""
    if sample.startswith("ENSG"):
        key_format = "ensembl_gene_id"
    elif sample.isupper() and sample.isalnum():
        key_format = "gene_symbol"
    else:
        key_format = "unknown"

    info = {
        "n_genes": len(keys),
        "embedding_dim": int(arr.shape[-1]) if arr.ndim else 0,
        "sample_keys": sample_keys,
        "key_format": key_format,
    }
    print(f"GenePT pickle: {pickle_path}")
    print(f"  n_genes        = {info['n_genes']}")
    print(f"  embedding_dim  = {info['embedding_dim']}")
    print(f"  key_format     = {info['key_format']}")
    print(f"  sample_keys    = {sample_keys}")
    return info


def load_genept_embeddings(pickle_path: str | Path) -> dict[str, np.ndarray]:
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}


def load_genept_summaries(summaries_path: str | Path) -> dict[str, str]:
    """Load NCBI summaries. Accepts JSON dict or pickle dict keyed by gene."""
    p = Path(summaries_path)
    if p.suffix == ".json":
        import json
        with open(p) as f:
            return json.load(f)
    with open(p, "rb") as f:
        return pickle.load(f)


def load_hgnc_family(group_id: int, short_name: str, cache_dir: str | Path) -> pd.DataFrame:
    """Download (or load from cache) an HGNC gene group as a DataFrame."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"hgnc_{group_id}.tsv"

    if not cache_file.exists():
        url = HGNC_URL.format(group_id=group_id)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        cache_file.write_text(r.text)

    df = pd.read_csv(cache_file, sep="\t")
    # HGNC TSV columns include: 'Approved symbol', 'Ensembl gene ID', ...
    sym_col = next((c for c in df.columns if "symbol" in c.lower() and "approved" in c.lower()), None)
    ens_col = next((c for c in df.columns if "ensembl" in c.lower() and "gene" in c.lower()), None)
    if sym_col is None or ens_col is None:
        raise ValueError(f"Unexpected HGNC columns: {list(df.columns)}")

    out = df[[sym_col, ens_col]].rename(columns={sym_col: "symbol", ens_col: "ensembl_id"})
    out = out.dropna(subset=["symbol"]).copy()
    out["family"] = short_name
    return out


def build_gene_table(
    genept_pickle: str | Path,
    summaries_path: str | Path,
    hgnc_cache: str | Path,
    families: list[tuple[int, str, str]] | None = None,
    per_family_limit: int | None = None,
) -> pd.DataFrame:
    """Join HGNC families with GenePT coverage. Returns a single DataFrame."""
    families = families or FAMILIES

    embeddings = load_genept_embeddings(genept_pickle)
    summaries = load_genept_summaries(summaries_path)
    embed_keys = set(embeddings.keys())

    rows = []
    for group_id, short_name, display_name in families:
        fam = load_hgnc_family(group_id, short_name, hgnc_cache)
        # GenePT is typically keyed by gene symbol; fall back to ensembl_id if needed.
        fam = fam[fam["symbol"].isin(embed_keys) | fam["ensembl_id"].isin(embed_keys)].copy()
        if per_family_limit:
            fam = fam.head(per_family_limit)
        print(f"  {display_name:25s}: {len(fam):4d} genes (HGNC ∩ GenePT)")
        for _, r in fam.iterrows():
            key = r["symbol"] if r["symbol"] in embed_keys else r["ensembl_id"]
            rows.append({
                "symbol": r["symbol"],
                "ensembl_id": r["ensembl_id"],
                "family": short_name,
                "summary": summaries.get(key, ""),
                "y_embedding": embeddings[key],
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["ensembl_id"]).drop_duplicates(subset=["ensembl_id"])
    return df.reset_index(drop=True)
