"""Load GenePT artifacts, HGNC complete set, and join them into a gene table.

We use the HGNC complete TSV (one row per gene, with a `gene_group` text column
that lists every gene group the gene belongs to) and substring-match against
that field per family. This is more robust than chasing single group IDs, since
HGNC has no master "all kinases" / "all TFs" group — only many small subgroups.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Each family: (short_name, display_name, [include_regexes], [exclude_regexes])
# Regexes are matched (case-insensitive) against the HGNC `gene_group` text,
# which is a "|"-separated list of group names per gene. Excludes win.
FAMILIES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "kinase", "Protein kinases",
        [r"\bkinase"],
        [r"kinase inhibitor", r"kinase regulator", r"kinase substrate", r"pseudokinase"],
    ),
    (
        "tf", "Transcription factors",
        [
            r"zinc finger", r"homeobox", r"basic helix-loop-helix", r"\bbZIP\b",
            r"forkhead box", r"high mobility group", r"nuclear receptor",
            r"T-box", r"SOX transcription", r"ETS transcription",
            r"transcription factor",
        ],
        [r"transcription factor binding", r"cofactor"],
    ),
    (
        "ion", "Ion channels",
        [r"\bchannel"],
        [r"channel regulat", r"channel auxiliary", r"channel interacting"],
    ),
    (
        "gpcr", "GPCRs",
        [
            r"G protein-coupled receptor", r"adrenoceptor",
            r"5-hydroxytryptamine receptor", r"dopamine receptor",
            r"muscarinic", r"opioid receptor", r"chemokine receptor",
            r"olfactory receptor",
        ],
        [],
    ),
    (
        "immune", "Immune receptors",
        [
            r"toll like receptor",
            r"interleukin .* receptor",
            r"\bFc receptor",
            r"NOD-like receptor", r"NLR family",
            r"killer cell immunoglobulin",
            r"T cell receptor", r"B cell receptor",
            r"C-type lectin domain",
            r"immunoglobulin like receptor",
        ],
        [r"binding"],
    ),
]

HGNC_COMPLETE_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)


def analyze_genept(pickle_path: str | Path) -> dict:
    """Print and return basic structure info for a GenePT embedding pickle."""
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict-like pickle, got {type(obj).__name__}")

    keys = list(obj.keys())
    sample_keys = keys[:10]
    arr = np.asarray(obj[keys[0]])
    sample = sample_keys[0] if sample_keys else ""
    if sample.startswith("ENSG"):
        key_format = "ensembl_gene_id"
    elif sample.isupper() and sample.replace("-", "").isalnum():
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
    p = Path(summaries_path)
    if p.suffix == ".json":
        import json
        with open(p) as f:
            return json.load(f)
    with open(p, "rb") as f:
        return pickle.load(f)


def load_hgnc_complete(cache_dir: str | Path) -> pd.DataFrame:
    """Download (once, then cached) the full HGNC TSV. Returns symbol/ensembl/gene_group."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "hgnc_complete_set.tsv"

    if not cache_file.exists():
        print(f"  downloading HGNC complete set -> {cache_file}")
        r = requests.get(HGNC_COMPLETE_URL, timeout=120)
        r.raise_for_status()
        cache_file.write_bytes(r.content)

    df = pd.read_csv(cache_file, sep="\t", low_memory=False, dtype=str)
    keep = ["symbol", "ensembl_gene_id", "gene_group", "locus_group"]
    df = df[keep].rename(columns={"ensembl_gene_id": "ensembl_id"})
    df["gene_group"] = df["gene_group"].fillna("")
    # protein-coding only — we want CDS sequences
    df = df[df["locus_group"] == "protein-coding gene"]
    return df.reset_index(drop=True)


def filter_family(
    hgnc: pd.DataFrame,
    includes: list[str],
    excludes: list[str],
) -> pd.DataFrame:
    inc_re = re.compile("|".join(includes), re.IGNORECASE) if includes else None
    exc_re = re.compile("|".join(excludes), re.IGNORECASE) if excludes else None
    if inc_re is None:
        return hgnc.iloc[0:0]
    mask = hgnc["gene_group"].str.contains(inc_re, na=False)
    if exc_re is not None:
        mask &= ~hgnc["gene_group"].str.contains(exc_re, na=False)
    return hgnc[mask].copy()


def build_gene_table(
    genept_pickle: str | Path,
    summaries_path: str | Path,
    hgnc_cache: str | Path,
    families: list | None = None,
    per_family_limit: int | None = None,
) -> pd.DataFrame:
    """Join HGNC families with GenePT coverage. Returns a single DataFrame."""
    families = families or FAMILIES

    embeddings = load_genept_embeddings(genept_pickle)
    summaries = load_genept_summaries(summaries_path)
    embed_keys = set(embeddings.keys())

    hgnc = load_hgnc_complete(hgnc_cache)

    rows = []
    seen_ensembl: set[str] = set()
    for short_name, display_name, includes, excludes in families:
        fam = filter_family(hgnc, includes, excludes)
        fam = fam.dropna(subset=["ensembl_id"])
        fam = fam[fam["symbol"].isin(embed_keys)]
        fam = fam[~fam["ensembl_id"].isin(seen_ensembl)]
        if per_family_limit:
            fam = fam.head(per_family_limit)
        seen_ensembl.update(fam["ensembl_id"].tolist())
        print(f"  {display_name:25s}: {len(fam):4d} genes (after HGNC ∩ GenePT ∩ ensembl)")
        for _, r in fam.iterrows():
            sym = r["symbol"]
            rows.append({
                "symbol": sym,
                "ensembl_id": r["ensembl_id"],
                "family": short_name,
                "summary": summaries.get(sym, ""),
                "y_embedding": embeddings[sym],
            })

    return pd.DataFrame(rows).reset_index(drop=True)
