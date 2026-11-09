"""CDS -> protein translation (standard genetic code, no external deps).

Used by the homology-aware clustering pipeline (translate CDS before MMseqs2
protein clustering) and by the translated amino-acid composition baselines.
A hardcoded codon table keeps this dependency-free; behaviour matches the
standard NCBI translation table 1.
"""
from __future__ import annotations

# Standard genetic code (NCBI translation table 1). '*' = stop.
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# The 20 standard amino acids, fixed order (used by AA-composition baselines).
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Any codon containing a non-ACGT base translates to this placeholder.
UNKNOWN_AA = "X"


MODES = ("through", "first_stop")
STOPS = frozenset(c for c, aa in CODON_TABLE.items() if aa == "*")


def translate_cds(seq: str, *, mode: str) -> str:
    """Translate a coding DNA sequence to its protein string, frame 0.

    ``mode`` is required, so every caller states what happens at a stop:

    - ``"through"``: full length. Internal stops become ``UNKNOWN_AA`` ('X') and
      a terminal stop is dropped. This is what Ensembl's own peptide holds (with
      '*' for X) and what the protein comparators use (G5).
    - ``"first_stop"``: halt at the first stop. Kept only for the MMseqs2
      clustering behind the frozen ``data/splits.json``; changing it would be a
      re-split.

    A trailing 1-2 nt remainder is dropped; codons with a non-ACGT base map to
    'X'. CDSs that break the rules are listed by :func:`check_translation`.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    s = seq.upper()
    codons = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
    out: list[str] = []
    for i, codon in enumerate(codons):
        aa = CODON_TABLE.get(codon, UNKNOWN_AA)
        if aa == "*":
            if mode == "first_stop" or i == len(codons) - 1:
                break
            aa = UNKNOWN_AA
        out.append(aa)
    return "".join(out)


def check_translation(seq: str) -> list[str]:
    """Reasons a CDS breaks "protein length = CDS/3 - 1, no internal stop".

    Returns ``[]`` for a clean CDS. The genes that fail are listed in
    ``data/translation_exceptions.tsv``; tests/test_translation.py checks the
    list against the CDS cache.
    """
    s = seq.upper()
    codons = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
    reasons = []
    if len(s) % 3:
        reasons.append("length_not_multiple_of_3")
    if any(c in STOPS for c in codons[:-1]):
        reasons.append("internal_stop")
    if not codons or codons[-1] not in STOPS:
        reasons.append("no_terminal_stop")
    if set(s) - set("ACGT"):
        reasons.append("non_acgt")
    return reasons
