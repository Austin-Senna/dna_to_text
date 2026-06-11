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


def translate_cds(seq: str, to_stop: bool = True) -> str:
    """Translate a coding DNA sequence to its protein string.

    - Reads in frame 0, codon by codon. A trailing 1-2 nt remainder is dropped.
    - Codons with any non-ACGT base map to ``UNKNOWN_AA`` ('X').
    - With ``to_stop=True`` (default) translation halts at the first stop codon
      and the stop is not included; otherwise stops are emitted as '*'.
    """
    s = seq.upper()
    out: list[str] = []
    for i in range(0, len(s) - len(s) % 3, 3):
        codon = s[i:i + 3]
        aa = CODON_TABLE.get(codon, UNKNOWN_AA)
        if aa == "*":
            if to_stop:
                break
            out.append(aa)
        else:
            out.append(aa)
    return "".join(out)
