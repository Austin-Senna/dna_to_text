"""Reader-facing names for the pooling variants.

Code, data files, and metrics keys keep the short identifiers (``meanG``,
``tss_nt_v2_meanmean``); the paper tables and figures show these names instead.
"""
from __future__ import annotations

POOL_DISPLAY = {
    "meanmean": "Mean",
    "specialmean": "Mean (Boundary-including)",
    "maxmean": "Max",
    "clsmean": "Mean-CLS",
    "meanD": "Ends + Mean",
    "meanG": "Ends + Mean + Max",
    "tssanchored": "TSS-Anchored",
}


def display_label(label: str) -> str:
    """'DNABERT-2 meanD (best DNA)' -> 'DNABERT-2, Ends + Mean (best DNA)'."""
    words = label.split(" ")
    for i, w in enumerate(words):
        if w in POOL_DISPLAY:
            words[i - 1] += ","
            words[i] = POOL_DISPLAY[w]
    return " ".join(words)
