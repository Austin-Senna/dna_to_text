"""Homology clustering of genes via MMseqs2 on translated CDS.

Pipeline: translate each gene's CDS to protein -> write a FASTA -> run
``mmseqs easy-cluster`` at a stated identity/coverage threshold -> parse the
``*_cluster.tsv`` into an ``ensembl_id -> cluster_id`` mapping. The mapping
feeds the homology-aware (whole-cluster) train/val/test split required by the
Bioinformatics submission, so paralogous genes cannot straddle the split.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from data_loader.sequence_fetcher import fetch_cds
from protein import translate_cds

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_DIR = REPO_ROOT / "data" / "sequences"


def write_protein_fasta(
    ids: list[str],
    out_fasta: Path,
    sequences_dir: Path = SEQUENCES_DIR,
) -> tuple[int, list[str]]:
    """Translate each gene's cached CDS and write a protein FASTA.

    Returns (n_written, missing_ids). A gene is "missing" if its CDS is not
    cached or translates to an empty protein.
    """
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    missing: list[str] = []
    with out_fasta.open("w") as f:
        for eid in ids:
            seq = fetch_cds(eid, sequences_dir)
            if not seq:
                missing.append(eid)
                continue
            protein = translate_cds(seq, to_stop=True)
            if not protein:
                missing.append(eid)
                continue
            f.write(f">{eid}\n{protein}\n")
            n_written += 1
    return n_written, missing


def run_mmseqs_cluster(
    fasta: Path,
    workdir: Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    cov_mode: int = 0,
    mmseqs_bin: str = "mmseqs",
) -> Path:
    """Run ``mmseqs easy-cluster`` and return the path to ``*_cluster.tsv``.

    cov_mode 0 = coverage of query and target (bidirectional), the standard
    choice for redundancy reduction.
    """
    if shutil.which(mmseqs_bin) is None:
        raise FileNotFoundError(
            f"mmseqs binary {mmseqs_bin!r} not found on PATH; install MMseqs2"
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    res_prefix = workdir / "res"
    tmp_dir = workdir / "tmp"
    cmd = [
        mmseqs_bin, "easy-cluster", str(fasta), str(res_prefix), str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    tsv = workdir / "res_cluster.tsv"
    if not tsv.exists():
        raise RuntimeError(f"mmseqs produced no cluster tsv at {tsv}")
    return tsv


def parse_cluster_tsv(tsv: Path) -> dict[str, str]:
    """Parse a representative<TAB>member cluster TSV into member -> rep id."""
    mapping: dict[str, str] = {}
    for line in Path(tsv).read_text().splitlines():
        if not line.strip():
            continue
        rep, member = line.split("\t")[:2]
        mapping[member] = rep
    return mapping


def cluster_genes(
    ids: list[str],
    workdir: Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    sequences_dir: Path = SEQUENCES_DIR,
    mmseqs_bin: str = "mmseqs",
) -> tuple[dict[str, str], dict]:
    """Cluster the given genes by translated-protein identity.

    Returns (ensembl_id -> cluster_id, stats). Genes whose CDS is missing are
    each assigned a unique singleton cluster id (``singleton:<id>``) so the
    downstream split still covers every gene.
    """
    workdir = Path(workdir)
    fasta = workdir / "proteins.fasta"
    n_written, missing = write_protein_fasta(ids, fasta, sequences_dir=sequences_dir)
    tsv = run_mmseqs_cluster(
        fasta, workdir, min_seq_id=min_seq_id, coverage=coverage, mmseqs_bin=mmseqs_bin
    )
    mapping = parse_cluster_tsv(tsv)
    for eid in missing:
        mapping[eid] = f"singleton:{eid}"
    # Any clustered gene absent from the tsv (shouldn't happen) -> singleton.
    for eid in ids:
        mapping.setdefault(eid, f"singleton:{eid}")
    n_clusters = len(set(mapping.values()))
    stats = {
        "n_genes": len(ids),
        "n_translated": n_written,
        "n_missing_cds": len(missing),
        "n_clusters": n_clusters,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
    }
    return mapping, stats


def cluster_dataframe(
    df: pd.DataFrame,
    workdir: Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    sequences_dir: Path = SEQUENCES_DIR,
    mmseqs_bin: str = "mmseqs",
) -> tuple[pd.DataFrame, dict]:
    """Add a ``cluster_id`` column to df (must have ``ensembl_id``)."""
    ids = df["ensembl_id"].tolist()
    mapping, stats = cluster_genes(
        ids, workdir, min_seq_id=min_seq_id, coverage=coverage,
        sequences_dir=sequences_dir, mmseqs_bin=mmseqs_bin,
    )
    out = df.copy()
    out["cluster_id"] = out["ensembl_id"].map(mapping)
    return out, stats
