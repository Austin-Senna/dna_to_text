"""All-vs-all protein search behind the evaluation purge (G2).

Translates every gene's CDS at full length (``translate_cds(mode="through")``,
the proteins the comparators use, G5), searches all against all with MMseqs2
with no prefilter, and keeps every unordered pair passing Rule A at 40%
identity: E <= 1e-3, identity >= 0.4, coverage >= 0.8 on both sequences
(measurements_2026-09.md §1). A pair passes when either search direction
passes; for leakage that is the conservative side. The row kept is the
passing direction with the higher identity.

The table is split-independent; ``splits.leaks.purge_for`` derives each
split's masked genes from it, so a mask can never fall out of step with its
split file.

Outputs (tracked):
  data/leaks/protein_pairs.tsv   gene_a, gene_b (gene_a < gene_b), fident, alnlen, qcov, tcov, evalue
  data/leaks/protein_pairs.json  inputs, MMseqs2 version and flags, counts

Run: uv run scripts/build_protein_pairs.py [--mmseqs ~/.local/bin/mmseqs] [--check]
  --check rebuilds the table and fails if it differs from the tracked one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from data_loader.sequence_fetcher import fetch_cds
from protein import translate_cds

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
UNIVERSE = DATA / "splits.json"
SEQUENCES = DATA / "sequences"
OUT = DATA / "leaks" / "protein_pairs.tsv"
META = DATA / "leaks" / "protein_pairs.json"
MMSEQS_VERSION = "18cc7493f392b95699ded8c0534dad1558ccc0f1"
SEARCH_FLAGS = ["--exhaustive-search", "1", "--max-seqs", "10000", "-e", "1e-3",
                "--seq-id-mode", "0", "--alignment-mode", "3", "--threads", "1",
                "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue"]
MAX_EVALUE, MIN_ID, MIN_COV = 1e-3, 0.40, 0.80
COLUMNS = ["gene_a", "gene_b", "fident", "alnlen", "qcov", "tcov", "evalue"]


class StalePairs(RuntimeError):
    """The tracked pair table differs from a rebuild."""


class WrongMMseqs(RuntimeError):
    """The MMseqs2 build is not the pinned one."""


def proteins() -> str:
    genes = sorted(g for s in ("train", "val", "test") for g in json.loads(UNIVERSE.read_text())[s])
    out = []
    for g in genes:
        cds = fetch_cds(g, SEQUENCES)
        if not cds:
            raise RuntimeError(f"no cached CDS for {g}")
        protein = translate_cds(cds, mode="through")
        out.append(f">{g}\n{protein}\n")
    return "".join(out)


def search(fasta_text: str, mmseqs: str) -> list[dict]:
    version = subprocess.run([mmseqs, "version"], capture_output=True, text=True, check=True).stdout.strip()
    if version != MMSEQS_VERSION:
        raise WrongMMseqs(f"{mmseqs} is {version}, expected {MMSEQS_VERSION}")
    with tempfile.TemporaryDirectory() as tmp:
        fasta, m8 = Path(tmp) / "proteins.fasta", Path(tmp) / "hits.m8"
        fasta.write_text(fasta_text)
        subprocess.run([mmseqs, "easy-search", str(fasta), str(fasta), str(m8), str(Path(tmp) / "work"),
                        *SEARCH_FLAGS], check=True, capture_output=True, text=True)
        rows = list(csv.reader(m8.open(), delimiter="\t"))
    return [dict(zip(["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue"], r)) for r in rows]


def rule_a_pairs(hits: list[dict]) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for h in hits:
        q, t = h["query"], h["target"]
        if q == t:
            continue
        fid, qcov, tcov, ev = float(h["fident"]), float(h["qcov"]), float(h["tcov"]), float(h["evalue"])
        if ev > MAX_EVALUE or fid < MIN_ID or qcov < MIN_COV or tcov < MIN_COV:
            continue
        a, b = sorted((q, t))
        # Coverage is reported for the query; orient it to (gene_a, gene_b).
        cov_a, cov_b = (qcov, tcov) if q == a else (tcov, qcov)
        row = {"gene_a": a, "gene_b": b, "fident": h["fident"], "alnlen": h["alnlen"],
               "qcov": f"{cov_a:.3f}", "tcov": f"{cov_b:.3f}", "evalue": h["evalue"]}
        key = (a, b)
        if key not in best or float(row["fident"]) > float(best[key]["fident"]):
            best[key] = row
    return [best[k] for k in sorted(best)]


def render(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmseqs", default=shutil.which("mmseqs") or os.path.expanduser("~/.local/bin/mmseqs"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    fasta_text = proteins()
    rows = rule_a_pairs(search(fasta_text, args.mmseqs))
    table = render(rows)
    if args.check:
        if not OUT.exists() or OUT.read_text() != table:
            raise StalePairs(f"{OUT.relative_to(REPO_ROOT)} differs from a rebuild")
        print(f"{OUT.relative_to(REPO_ROOT)} matches its rebuild ({len(rows)} pairs)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(table)
    META.write_text(json.dumps({
        "proteins": {"translation": 'translate_cds(mode="through")',
                     "n": fasta_text.count(">"),
                     "fasta_sha256": hashlib.sha256(fasta_text.encode()).hexdigest(),
                     "gene_universe": {"path": UNIVERSE.relative_to(REPO_ROOT).as_posix(),
                                       "sha256": hashlib.sha256(UNIVERSE.read_bytes()).hexdigest()}},
        "mmseqs": {"version": MMSEQS_VERSION, "command": "easy-search", "flags": SEARCH_FLAGS},
        "rule_a": {"max_evalue": MAX_EVALUE, "min_fident": MIN_ID, "min_coverage_both": MIN_COV,
                   "direction": "either direction passing"},
        "n_pairs": len(rows),
        "table_sha256": hashlib.sha256(table.encode()).hexdigest(),
    }, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(rows)} pairs")


if __name__ == "__main__":
    main()
