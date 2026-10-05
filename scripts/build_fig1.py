#!/usr/bin/env python3
"""Figure 1's numbers, written into its draw.io source, and the PNG rendered from it.

The study-design figure (``dna_to_text_paper/paper/figures/dna_to_text_detailed.drawio``)
is Austin's hand-drawn diagram. Its panel A gene counts and chunk lengths, panel B
chunk settings, and panel C scores are typed text; this
builder owns those cells (``CELLS``, located by cell id), fills them from
``numbers.tex`` (the values the prose prints) and the encoder registry (the
extraction settings), renders ``mina_fig1.png`` with draw.io's headless image (pinned,
byte-deterministic), and writes ``mina_fig1.stamp`` with the digests of both, so
a guard can tell when the PNG is older than the source. Everything else in the
drawing is edited by hand, then this builder re-renders it.

Run: uv run scripts/build_fig1.py            fill, and render if stale (needs Docker, sandbox off)
     uv run scripts/build_fig1.py --check    fail if a cell, a typed decimal or the PNG is stale
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

from data_loader.model_registry import ENCODER_SPECS

PAPER = Path(__file__).resolve().parents[1] / "dna_to_text_paper" / "paper"
NUMBERS = PAPER / "numbers.tex"
DRAWIO = PAPER / "figures" / "dna_to_text_detailed.drawio"
PNG = PAPER / "figures" / "mina_fig1.png"
STAMP = PAPER / "figures" / "mina_fig1.stamp"
IMAGE = "rlespinasse/drawio-desktop-headless@sha256:f33bc2f204738209a063ce38edf8003959c3be09cc18ecc9087a295aa5c585ef"
EXPORT_SCALE = "0.95"   # the headless export hangs (exit 124 after its 10 s timeout) on a PNG wider than about 1,760 px; the drawing is about 1,840

# Cell id -> its value, with {key} slots read from numbers.tex or, for chunk.* and
# overlap.*, from the encoder registry (``registry_values``).
CELLS = {
    "434": "<b>({n.genes} total)</b>",
    "436": "{n.family.tf}",
    "438": "{n.family.gpcr}",
    "440": "{n.family.kinase}",
    "442": "{n.family.ion}",
    "444": "{n.family.immune}",
    "460": "<b>DNABERT-2</b><br><span style='font-size:13px;color:#111827'>W = {chunk.dnabert2}, D = 768</span>",
    "487": "Use encoder-compatible W<br>example: W = {chunk.dnabert2} tokens",
    "489": "c<sub>2</sub> … c<sub>K</sub> (longer genes)<br>overlap: {overlap.dnabert2} tokens",
    "538": "<div>{cds.f5.best-encoder}, {cds.f5.best-encoder.pool}: {cds.f5.best-encoder.value}</div>"
           "<div>AA {cds.f5.aa-kmer.k}-mer: {cds.f5.aa-kmer}</div>"
           "<div>CDS {cds.f5.nt-kmer.k}-mer: {cds.f5.nt-kmer}</div>",
    "664": "<div>{cds.gp.best-encoder}, {cds.gp.best-encoder.pool}: {cds.gp.best-encoder.value}</div>"
           "<div>AA {cds.gp.aa-kmer.k}-mer: {cds.gp.aa-kmer}</div>"
           "<div>CDS {cds.gp.nt-kmer.k}-mer: {cds.gp.nt-kmer}</div>",
    "619": "{chunk.dnabert2}",
    "622": "{chunk.nt-v2}",
    "625": "{chunk.gena-lm}",
    "628": "{chunk.hyena-dna}",
}
EXAMPLE_EMBEDDING = "574"     # panel B's e_c1: HTR1A's one DNABERT-2 chunk, token mean, first two dims (tests check the cache)
SHORT_POOL = {"Mean (Boundary-including)": "Mean (Bound.-incl.)"}   # as panel B's pooling list spells it
NAMEDEF = re.compile(r"\\@namedef\{mina@([^}]*)\}\{(.*)\}$")
SLOT = re.compile(r"\{([a-z0-9.-]+)\}")
DECIMAL = re.compile(r"\d+\.\d+")


def read_numbers(path: Path = NUMBERS) -> dict[str, str]:
    return {m[1]: m[2] for line in path.read_text().splitlines() if (m := NAMEDEF.match(line))}


def plain(tex: str) -> str:
    """A numbers.tex value as figure text; refuses what the figure cannot print."""
    if r"\sens" in tex:
        raise ValueError(f"{tex!r} carries a dagger (its digits move under a refit); the figure has no footnote for it")
    text = tex.replace("{,}", ",").replace(r"\ensuremath{-}", "\u2212")
    if re.search(r"[\\{}]", text):
        raise ValueError(f"{tex!r}: TeX the figure cannot print")
    return text


def registry_values() -> dict[str, str]:
    """Each encoder's CDS chunk length W in content tokens (boundary tokens excluded, as
    the appendix states it) and the overlap between chunks, as numbers.tex-style values."""
    return {f"{field}.{name.replace('_', '-')}": f"{n:,}".replace(",", "{,}")
            for name, spec in ENCODER_SPECS.items()
            for field, n in (("chunk", spec.max_content_tokens), ("overlap", spec.stride))}


def cell_values(numbers: dict[str, str]) -> dict[str, str]:
    known = {**numbers, **registry_values()}

    def slot(m: re.Match) -> str:
        text = plain(known[m[1]])
        return SHORT_POOL.get(text, text) if m[1].endswith(".pool") else text
    return {cell: SLOT.sub(slot, template) for cell, template in CELLS.items()}


def fill(text: str, values: dict[str, str]) -> str:
    """``text`` with each cell's value attribute replaced; the rest of the file untouched."""
    for cell, value in values.items():
        attr = escape(value, {'"': "&quot;", "\n": "&#10;"})
        text, n = re.subn(rf'(<mxCell id="{cell}" value=")[^"]*"', lambda m: f'{m[1]}{attr}"', text)
        if n != 1:
            raise ValueError(f"cell {cell}: {n} matches in {DRAWIO.name}")
    return text


def values_in(text: str) -> dict[str, str]:
    return {c.get("id"): c.get("value", "") for c in ET.fromstring(text).iter("mxCell")}


def stale_cells(text: str, numbers: dict[str, str]) -> list[tuple[str, str | None, str]]:
    have = values_in(text)
    return [(cell, have.get(cell), want) for cell, want in cell_values(numbers).items() if have.get(cell) != want]


def stray_decimals(text: str) -> list[tuple[str, str]]:
    """Decimals typed into a cell the builder does not fill (tags stripped, so styles don't count)."""
    return [(cell, d) for cell, value in values_in(text).items() if cell not in {*CELLS, EXAMPLE_EMBEDDING}
            for d in DECIMAL.findall(html.unescape(re.sub(r"<[^>]+>", "", value)))]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp_for(drawio: bytes, png: bytes) -> dict:
    return {"drawio": DRAWIO.name, "drawio_sha256": _sha(drawio), "png": PNG.name, "png_sha256": _sha(png),
            "renderer": IMAGE, "export_scale": EXPORT_SCALE}


def stamp_problems(drawio: bytes, png: bytes | None, stamp: dict | None) -> list[str]:
    if stamp is None or png is None:
        return [f"no {STAMP.name if stamp is None else PNG.name}: run scripts/build_fig1.py"]
    want = stamp_for(drawio, png)
    problems = {"drawio_sha256": f"{DRAWIO.name} changed since {PNG.name} was rendered",
                "png_sha256": f"{PNG.name} is not the render the stamp records",
                "renderer": f"{PNG.name} was rendered by {stamp.get('renderer')}, not {IMAGE}",
                "export_scale": f"{PNG.name} was rendered at scale {stamp.get('export_scale')}, not {EXPORT_SCALE}"}
    return [msg for field, msg in problems.items() if stamp.get(field) != want[field]]


def _on_disk() -> tuple[bytes, bytes | None, dict | None]:
    return (DRAWIO.read_bytes(), PNG.read_bytes() if PNG.exists() else None,
            json.loads(STAMP.read_text()) if STAMP.exists() else None)


def render(drawio: bytes) -> bytes:
    """The PNG the pinned headless draw.io renders from ``drawio``, cropped to the drawing."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, DRAWIO.name).write_bytes(drawio)
        subprocess.run(["docker", "run", "--rm", "-w", "/data", "-v", f"{tmp}:/data", IMAGE,
                        "-x", "-f", "png", "--crop", "-s", EXPORT_SCALE, "-o", PNG.name, DRAWIO.name],
                       check=True, timeout=600)
        return Path(tmp, PNG.name).read_bytes()


def check(numbers: dict[str, str]) -> list[str]:
    drawio, png, stamp = _on_disk()
    text = drawio.decode()
    problems = [f"cell {c} prints {have!r}, numbers.tex gives {want!r}" for c, have, want in stale_cells(text, numbers)]
    problems += [f"cell {c} types the decimal {d}: give it a slot in CELLS or drop it" for c, d in stray_decimals(text)]
    return problems + stamp_problems(drawio, png, stamp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if a cell or the PNG is stale; writes nothing")
    args = ap.parse_args()
    numbers = read_numbers()
    if not args.check:
        drawio = DRAWIO.read_bytes()
        filled = fill(drawio.decode(), cell_values(numbers)).encode()
        if filled != drawio:
            DRAWIO.write_bytes(filled)
        _, png, stamp = _on_disk()
        if stamp_problems(filled, png, stamp):
            png = render(filled)
            PNG.write_bytes(png)
            STAMP.write_text(json.dumps(stamp_for(filled, png), indent=2) + "\n")
            print(f"rendered {PNG.name}")
    problems = check(numbers)
    for p in problems:
        print(f"  {p}")
    print(f"{DRAWIO.name}: {len(CELLS)} cells from numbers.tex and the encoder registry, {len(problems)} problems")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
