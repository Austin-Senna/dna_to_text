"""Build the strand-aware canonical-TSS windows and their manifest (G4, G19).

For every gene in the split universe:
1. The TSS is the 5' end of the transcript whose CDS we embed (the versioned
   ENST in the ``data/sequences/{gene}.fa`` header), with coordinates from the
   pinned Ensembl GTF (``enformer_windows.GTF_PATH``). Every transcript version
   must be in that release.
2. The window puts the TSS at ``TSS_INDEX`` in gene orientation
   (``enformer_windows.window_span``/``orient``): forward-strand bases from the
   same release's primary-assembly FASTA, reverse-complemented on the minus
   strand, N-padded at a chromosome edge.
3. Checks, all of which must pass before the manifest is written:
   - G4: the window from ``TSS_INDEX`` matches the first k bases of the
     transcript's cDNA from the release's cDNA FASTA, an independent source
     (k = min(50, exon-1 length)). The prefix is kept in the manifest so the
     check reruns offline.
   - Regression: wherever a new window's genomic span overlaps the May window
     (``data/enformer_windows``, forward strand, fetched from Ensembl REST),
     the bases are identical. Without the May cache this check has nothing to
     compare, so the build refuses unless ``--no-may-regression`` is passed.

Outputs: ``data/tss_windows_e{release}/{gene}.fa`` (gitignored),
``data/tss_windows.tsv`` (tracked manifest, one row per gene with the span, pads,
cDNA prefix and sha256) and ``data/tss_windows.meta.json`` (release, GTF sha256).
Every window is rebuilt from the genome on each run; an existing file with
different content raises instead of being overwritten or trusted.

Inputs (Ensembl FTP, release ``ENSEMBL_RELEASE``, in data/annotation/, gitignored): the GTF,
the cDNA FASTA, and the primary-assembly FASTA gunzipped once (``gunzip -k``).

Run: uv run scripts/build_tss_windows.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_loader.enformer_windows import (
    CDNA_FASTA,
    ENFORMER_WINDOW_LENGTH,
    ENSEMBL_RELEASE,
    GENOME_FASTA,
    GTF_PATH,
    MANIFEST,
    TSS_INDEX,
    WINDOW_DIR,
    FastaIndex,
    StaleWindow,
    WindowSpan,
    _parse_fasta,
    orient,
    read_cdna_prefixes,
    window_span,
    write_window,
)
from linear_trainer.sources import META_PARQUET

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OLD_CACHE = DATA / "enformer_windows"
CDNA_K = 50

_ATTR = {k: re.compile(rf'{k} "([^"]+)"') for k in
         ("gene_id", "transcript_id", "transcript_version", "exon_number")}


class WindowCheckFailed(RuntimeError):
    """A built window failed the cDNA check or the regression against May."""


def read_gtf(gtf: Path, transcripts: set[str]) -> dict[str, dict]:
    """transcript_id.version -> {chrom, start, end, strand, gene_id, exon1_len}."""
    out: dict[str, dict] = {}
    exon1: dict[str, int] = {}
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon"):
                continue
            tid = _ATTR["transcript_id"].search(f[8])
            ver = _ATTR["transcript_version"].search(f[8])
            if not tid or not ver:
                continue
            key = f"{tid.group(1)}.{ver.group(1)}"
            if key not in transcripts:
                continue
            if f[2] == "transcript":
                out[key] = {"chrom": f[0], "start": int(f[3]), "end": int(f[4]),
                            "strand": 1 if f[6] == "+" else -1,
                            "gene_id": _ATTR["gene_id"].search(f[8]).group(1)}
            elif _ATTR["exon_number"].search(f[8]).group(1) == "1":
                exon1[key] = int(f[4]) - int(f[3]) + 1
    for key, rec in out.items():
        rec["exon1_len"] = exon1[key]
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def old_window_diff(gene: str, span: WindowSpan, fwd: str, old_cache: Path) -> tuple[int, int] | None:
    """(bases compared, bases differing) against the May forward-strand window."""
    fa, lk = old_cache / f"{gene}.fa", old_cache / "_lookup" / f"{gene}.json"
    if not fa.exists() or not lk.exists():
        return None
    d = json.loads(lk.read_text())
    old_tss = int(d["start"]) if int(d.get("strand", 1)) >= 0 else int(d["end"])
    old_start = max(old_tss - ENFORMER_WINDOW_LENGTH // 2, 1)  # the May left-shift rule
    old = _parse_fasta(fa.read_text())
    if str(d["seq_region_name"]) != span.chrom:
        return (0, 0)
    a, b = max(old_start, span.start), min(old_start + len(old) - 1, span.end)
    if a > b:
        return (0, 0)
    x = old[a - old_start:b - old_start + 1]
    y = fwd[a - span.start:b - span.start + 1]
    if len(x) != len(y):
        raise WindowCheckFailed(f"{gene}: May window slice {len(x)} bp vs new {len(y)} bp")
    return (len(x), sum(p != q for p, q in zip(x, y)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", type=Path, default=GTF_PATH)
    ap.add_argument("--genome", type=Path, default=GENOME_FASTA,
                    help="uncompressed primary-assembly FASTA (gunzip -k the FTP file)")
    ap.add_argument("--cdna", type=Path, default=CDNA_FASTA)
    ap.add_argument("--splits", type=Path, default=DATA / "splits.json",
                    help="gene universe (train + val + test)")
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--window-dir", type=Path, default=WINDOW_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--old-cache", type=Path, default=OLD_CACHE)
    ap.add_argument("--max-genes", type=int, default=None,
                    help="pilot limit (needs a non-default --manifest and --window-dir)")
    ap.add_argument("--no-may-regression", action="store_true",
                    help="allow a build where no May window was available to compare")
    args = ap.parse_args()
    if args.max_genes is not None and (args.manifest.resolve() == MANIFEST.resolve()
                                       or args.window_dir.resolve() == WINDOW_DIR.resolve()):
        raise ValueError("--max-genes would overwrite the tracked manifest or the window cache")

    universe = json.loads(args.splits.read_text())
    genes = sorted(g for s in ("train", "val", "test") for g in universe[s])[:args.max_genes]
    transcript_of = {g: (args.seq_cache / f"{g}.fa").read_text().split("\n", 1)[0][1:].split()[0]
                     for g in genes}
    symbols = pd.read_parquet(META_PARQUET, columns=["ensembl_id", "symbol"])
    symbol_of = dict(zip(symbols["ensembl_id"], symbols["symbol"]))

    print(f"=== {len(genes)} genes; coordinates from {args.gtf.name} ===", flush=True)
    gtf = read_gtf(args.gtf, set(transcript_of.values()))
    absent = [g for g in genes if transcript_of[g] not in gtf]
    wrong = [g for g in genes if transcript_of[g] in gtf and gtf[transcript_of[g]]["gene_id"] != g]
    if absent or wrong:
        raise StaleWindow(f"release {ENSEMBL_RELEASE} lacks {len(absent)} CDS transcript "
                          f"versions {absent[:5]}; {len(wrong)} map to another gene {wrong[:5]}")
    if not args.genome.exists():
        raise FileNotFoundError(f"{args.genome} missing: gunzip -k {args.genome}.gz")
    genome = FastaIndex(args.genome)
    lengths = genome.lengths()
    cdna = read_cdna_prefixes(args.cdna, set(transcript_of.values()), CDNA_K)
    no_cdna = [g for g in genes if transcript_of[g] not in cdna]
    if no_cdna:
        raise StaleWindow(f"{len(no_cdna)} transcripts missing from {args.cdna.name}: {no_cdna[:5]}")

    rows, cdna_fail, diff_fail = [], [], []
    compared = n_bases = 0
    for g in tqdm(genes, desc="TSS windows"):
        t = gtf[transcript_of[g]]
        tss = t["start"] if t["strand"] > 0 else t["end"]
        span = window_span(t["chrom"], tss, t["strand"], lengths[t["chrom"]])
        row = {"symbol": symbol_of.get(g, ""), "transcript_id": transcript_of[g],
               "chrom": span.chrom, "strand": span.strand, "tss": span.tss,
               "start": span.start, "end": span.end, "pad_up": span.pad_up,
               "pad_down": span.pad_down, "chrom_len": lengths[t["chrom"]]}
        fwd = genome.fetch(span.chrom, span.start, span.end)
        seq = orient(fwd, span)
        path = args.window_dir / f"{g}.fa"
        if path.exists() and _parse_fasta(path.read_text()) != seq:
            raise StaleWindow(f"{path} differs from the window built now; use a fresh --window-dir")

        prefix = cdna[transcript_of[g]][:min(CDNA_K, t["exon1_len"])]
        if seq[TSS_INDEX:TSS_INDEX + len(prefix)] != prefix:
            cdna_fail.append(g)
        diff = old_window_diff(g, span, fwd, args.old_cache)
        if diff and diff[0]:
            compared += 1
            n_bases += diff[0]
            if diff[1]:
                diff_fail.append((g, diff[1]))
        rows.append(write_window(args.window_dir, g, seq,
                                 {**row, "exon1_len": t["exon1_len"], "cdna_prefix": prefix}))

    manifest = pd.DataFrame(rows).sort_values("ensembl_id")
    print(f"strand: {int((manifest['strand'] > 0).sum())} plus, "
          f"{int((manifest['strand'] < 0).sum())} minus; "
          f"N-padded at an edge: {int(((manifest['pad_up'] + manifest['pad_down']) > 0).sum())}")
    print(f"cDNA prefix at the window centre: {len(genes) - len(cdna_fail)}/{len(genes)} match")
    print(f"regression vs May windows: {compared} genes, {n_bases:,} overlapping bases, "
          f"{len(diff_fail)} genes differ")
    if cdna_fail or diff_fail:
        raise WindowCheckFailed(f"cDNA mismatch: {cdna_fail[:10]}; "
                                f"differs from May on shared bases: {diff_fail[:10]}")
    if compared == 0 and not args.no_may_regression:
        raise WindowCheckFailed(f"no May window in {args.old_cache} to compare against; "
                                "pass --no-may-regression to build without that check")

    manifest.to_csv(args.manifest, sep="\t", index=False)
    meta = {"ensembl_release": ENSEMBL_RELEASE, "assembly": "GRCh38",
            "window_length": ENFORMER_WINDOW_LENGTH, "tss_index": TSS_INDEX,
            "inputs": {p.name: _sha256(p) for p in (args.gtf, args.genome, args.cdna)},
            "cdna_check_bases": CDNA_K, "n_genes": len(manifest)}
    args.manifest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.manifest} ({len(manifest)} windows) and "
          f"{args.manifest.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
