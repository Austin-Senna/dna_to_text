"""Strand-aware, canonical-TSS genomic windows for every TSS encoder (G4, G19).

A window is exactly ``ENFORMER_WINDOW_LENGTH`` bases in gene orientation, with
the 5' end of the gene's canonical transcript (the transcript whose CDS we
embed) at index ``TSS_INDEX``: 98,304 bases upstream, the TSS, then 98,303
downstream. Minus-strand windows are reverse-complemented. Where a window runs
off a chromosome it is N-padded on that side, never shifted, so the TSS stays
at the same index for every gene.

Coordinates come from the pinned Ensembl GTF and sequence from the same
release's GRCh38 primary-assembly FASTA (release ``ENSEMBL_RELEASE``; Ensembl
REST was down on Sept 29, 2026, and the local FASTA is pinned and offline);
``scripts/build_tss_windows.py`` builds the window files and the tracked
manifest ``data/tss_windows.tsv``. Consumers read a
window only through :func:`read_window`, which checks it against the manifest's
sha256, so a file from an older geometry can never be picked up silently.

(The May 2026 windows centred forward-strand sequence on the gene's outermost
boundary; they stay in ``data/enformer_windows/`` for the Phase 3 replay of the
old code path, which runs at an older commit.)
"""
from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ENFORMER_WINDOW_LENGTH = 196_608
TSS_INDEX = ENFORMER_WINDOW_LENGTH // 2
# The release the CDS sequences were fetched from (REST served 115 in May 2026):
# all 3,244 CDS transcript versions are present, and canonical, in 115. Release
# 116 drops or re-versions 6 of them (NOBOX, NLRP5, NLRP8, OR4A8, OR1P1, ...),
# and 110 differs on 10; build_tss_windows.py refuses a release that lacks any.
ENSEMBL_RELEASE = 115

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
GTF_PATH = DATA / "annotation" / f"Homo_sapiens.GRCh38.{ENSEMBL_RELEASE}.gtf.gz"
WINDOW_DIR = DATA / f"tss_windows_e{ENSEMBL_RELEASE}"
MANIFEST = DATA / "tss_windows.tsv"
# Ensembl FTP files of that release; the genome is gunzipped once for random access.
GENOME_FASTA = DATA / "annotation" / f"Homo_sapiens.GRCh38.dna.primary_assembly.{ENSEMBL_RELEASE}.fa"
CDNA_FASTA = DATA / "annotation" / f"Homo_sapiens.GRCh38.cdna.all.{ENSEMBL_RELEASE}.fa.gz"

# IUPAC complements; GRCh38 carries a few ambiguity codes besides N.
_IUPAC_FROM = "ACGTNRYKMSWBDHV"
_IUPAC_TO = "TGCANYRMKSWVHDB"
_COMPLEMENT = str.maketrans(_IUPAC_FROM, _IUPAC_TO)


class StaleWindow(RuntimeError):
    """A window file is missing from the manifest or differs from it."""


@dataclass(frozen=True)
class WindowSpan:
    """The genomic bases a window covers and the N padding around them.

    ``start``/``end`` are 1-based inclusive forward-strand coordinates of the
    bases actually present; ``pad_up``/``pad_down`` count the Ns added upstream
    and downstream in gene orientation.
    """
    chrom: str
    strand: int
    tss: int
    start: int
    end: int
    pad_up: int
    pad_down: int
    length: int


def window_span(chrom: str, tss: int, strand: int, chrom_len: int,
                length: int = ENFORMER_WINDOW_LENGTH) -> WindowSpan:
    """The span that puts ``tss`` at index ``length // 2`` in gene orientation."""
    if length <= 0:
        raise ValueError("length must be positive")
    if strand not in (1, -1):
        raise ValueError(f"strand must be 1 or -1, got {strand!r}")
    if not 1 <= tss <= chrom_len:
        raise ValueError(f"TSS {tss} outside chromosome {chrom} (length {chrom_len})")
    up = length // 2          # bases upstream of the TSS
    down = length - up - 1    # bases downstream of it
    lo, hi = (tss - up, tss + down) if strand > 0 else (tss - down, tss + up)
    start, end = max(lo, 1), min(hi, chrom_len)
    clip_lo, clip_hi = start - lo, hi - end
    pad_up, pad_down = (clip_lo, clip_hi) if strand > 0 else (clip_hi, clip_lo)
    return WindowSpan(str(chrom), int(strand), int(tss), int(start), int(end),
                      int(pad_up), int(pad_down), int(length))


def reverse_complement(seq: str) -> str:
    bad = set(seq.upper()) - set(_IUPAC_FROM)
    if bad:
        raise ValueError(f"cannot complement {sorted(bad)}")
    return seq.upper().translate(_COMPLEMENT)[::-1]


def orient(forward_seq: str, span: WindowSpan) -> str:
    """Forward-strand bases of ``span`` -> the window in gene orientation."""
    seq = forward_seq.upper()
    if len(seq) != span.end - span.start + 1:
        raise ValueError(f"expected {span.end - span.start + 1} bases for "
                         f"{span.chrom}:{span.start}-{span.end}, got {len(seq)}")
    if span.strand < 0:
        seq = reverse_complement(seq)
    out = "N" * span.pad_up + seq + "N" * span.pad_down
    if len(out) != span.length:
        raise ValueError(f"window is {len(out)} bp, expected {span.length}")
    return out


def sha256_seq(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def _parse_fasta(text: str) -> str:
    return "".join(ln.strip() for ln in text.splitlines()
                   if ln and not ln.startswith(">")).upper()


HEADER_KEYS = ("transcript_id", "chrom", "start", "end", "strand", "tss", "pad_up", "pad_down")


def write_window(window_dir: str | Path, gene_id: str, seq: str, row: dict) -> dict:
    """Write one window file; return its manifest row (with ``n_N`` and ``sha256``)."""
    window_dir = Path(window_dir)
    window_dir.mkdir(parents=True, exist_ok=True)
    header = " ".join(f"{k}={row[k]}" for k in HEADER_KEYS)
    (window_dir / f"{gene_id}.fa").write_text(
        f">{gene_id} {header} release={ENSEMBL_RELEASE}\n{seq}\n")
    return {"ensembl_id": gene_id, **row, "n_N": seq.count("N"), "sha256": sha256_seq(seq)}


_MANIFESTS: dict[tuple, pd.DataFrame] = {}


def load_manifest(path: str | Path = MANIFEST) -> pd.DataFrame:
    """The window manifest, indexed by ``ensembl_id``."""
    path = Path(path)
    if not path.exists():
        raise StaleWindow(f"no window manifest at {path}; run scripts/build_tss_windows.py")
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _MANIFESTS:
        _MANIFESTS[key] = pd.read_csv(path, sep="\t", dtype={"chrom": str}).set_index("ensembl_id")
    return _MANIFESTS[key]


def read_window(gene_id: str, *, manifest: str | Path = MANIFEST,
                window_dir: str | Path = WINDOW_DIR,
                length: int = ENFORMER_WINDOW_LENGTH) -> str:
    """The gene's window, verified against the manifest (G19)."""
    m = load_manifest(manifest)
    if gene_id not in m.index:
        raise StaleWindow(f"{gene_id} is not in the window manifest {manifest}")
    path = Path(window_dir) / f"{gene_id}.fa"
    if not path.exists():
        raise StaleWindow(f"no window file {path}; run scripts/build_tss_windows.py")
    seq = _parse_fasta(path.read_text())
    if len(seq) != length or sha256_seq(seq) != m.at[gene_id, "sha256"]:
        raise StaleWindow(f"{path} does not match the manifest (length {len(seq)}, "
                          f"expected {length} with sha256 {m.at[gene_id, 'sha256'][:12]})")
    return seq


def window_spans(manifest: str | Path = MANIFEST) -> dict[str, tuple[str, int, int]]:
    """gene -> (chrom, start, end): the genomic bases each window covers."""
    m = load_manifest(manifest)
    return {g: (str(r.chrom), int(r.start), int(r.end)) for g, r in m.iterrows()}


# --- Local Ensembl FASTA ------------------------------------------------------

class FastaIndex:
    """Random access to an uncompressed FASTA with fixed line widths.

    The index is samtools' ``.fai`` layout (name, length, offset, bases per
    line, bytes per line), built on first use next to the FASTA.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        fai = self.path.with_name(self.path.name + ".fai")
        if not fai.exists():
            self._build(fai)
        self.index = {}
        for line in fai.read_text().splitlines():
            name, length, offset, lb, lw = line.split("\t")
            self.index[name] = (int(length), int(offset), int(lb), int(lw))

    def _build(self, fai: Path) -> None:
        rows, name, length, offset, lb, lw, short = [], None, 0, 0, 0, 0, False
        pos = 0
        with self.path.open("rb") as fh:
            for raw in fh:
                if raw.startswith(b">"):
                    if name is not None:
                        rows.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}")
                    name, length, offset, lb, lw, short = raw[1:].split()[0].decode(), 0, pos + len(raw), 0, 0, False
                else:
                    n = len(raw.rstrip(b"\r\n"))
                    if lb == 0:
                        lb, lw = n, len(raw)
                    elif short or n > lb:
                        raise ValueError(f"{self.path}: uneven line widths in {name}")
                    short = n < lb
                    length += n
                pos += len(raw)
        if name is not None:
            rows.append(f"{name}\t{length}\t{offset}\t{lb}\t{lw}")
        fai.write_text("\n".join(rows) + "\n")

    def lengths(self) -> dict[str, int]:
        return {name: v[0] for name, v in self.index.items()}

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Forward-strand bases chrom:start-end (1-based, inclusive)."""
        length, offset, lb, lw = self.index[chrom]
        if not 1 <= start <= end <= length:
            raise ValueError(f"{chrom}:{start}-{end} outside 1..{length}")
        a = offset + (start - 1) // lb * lw + (start - 1) % lb
        b = offset + (end - 1) // lb * lw + (end - 1) % lb
        with self.path.open("rb") as fh:
            fh.seek(a)
            raw = fh.read(b - a + 1)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()
        if len(seq) != end - start + 1:
            raise ValueError(f"{self.path}: read {len(seq)} bases for {chrom}:{start}-{end}")
        return seq


def read_cdna_prefixes(path: str | Path, transcripts: set[str], k: int) -> dict[str, str]:
    """First ``k`` cDNA bases (5' to 3') of each versioned transcript in ``transcripts``.

    The cDNA FASTA is the independent check on the window centre (G4): it is the
    transcript's own sequence, not a slice of the genome.
    """
    out: dict[str, str] = {}
    current, parts, have = None, [], 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if current is not None:
                    out[current] = "".join(parts)[:k].upper()
                tid = line[1:].split()[0]
                current, parts, have = (tid, [], 0) if tid in transcripts else (None, [], 0)
            elif current is not None and have < k:
                parts.append(line.strip())
                have += len(parts[-1])
    if current is not None:
        out[current] = "".join(parts)[:k].upper()
    return out
