"""Figure 1 (truth pass E): the study-design figure is Austin's hand-drawn draw.io
diagram, and its panel A counts, panel B single-window share and panel C scores
are typed text. The builder fills those cells from numbers.tex and stamps the
rendered PNG with the .drawio's digest; these guards fail when a number moves
without the figure, when the .drawio changes without a re-render, when a result
decimal sits in a cell the builder does not own, or when panel B's worked example
stops being HTR1A's real sequence, chunk and embedding."""
from __future__ import annotations

import dataclasses
import html
import json
import re

import numpy as np
import pytest

import build_fig1 as bf
from data_loader.model_registry import ENCODER_SPECS
from linear_trainer import records as R

HTR1A = "ENSG00000178394"


@pytest.fixture(scope="module")
def numbers():
    return bf.read_numbers()


@pytest.fixture(scope="module")
def drawio():
    return bf.DRAWIO.read_text()


def _text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value))


def test_the_drawio_prints_the_numbers_tex_values(numbers, drawio):
    assert bf.stale_cells(drawio, numbers) == []
    # Each line rebuilt from numbers.tex here, not through the builder's templates.
    cells = bf.values_in(drawio)
    for cell, task in (("538", "cds.f5"), ("664", "cds.gp")):
        def v(key):
            return bf.plain(numbers[f"{task}.{key}"])
        pool = v("best-encoder.pool").replace("Boundary-including", "Bound.-incl.")
        lines = [_text(div).strip() for div in re.findall(r"<div>(.*?)</div>", cells[cell])]
        assert lines == [f"{v('best-encoder')}, {pool}: {v('best-encoder.value')}",
                         f"AA {v('aa-kmer.k')}-mer: {v('aa-kmer')}",
                         f"CDS {v('nt-kmer.k')}-mer: {v('nt-kmer')}"], (cell, lines)
    assert _text(cells["436"]) == bf.plain(numbers["n.family.tf"])
    assert bf.plain(numbers["n.genes"]) in _text(cells["434"])
    assert bf.plain(numbers["single-chunk.dnabert2.pct"]) in _text(cells["489"])


def test_the_worked_example_is_htr1as_real_sequence(drawio):
    """Panel B prints HTR1A's CDS ends and length, and its one chunk's ends."""
    fasta = R.REPO_ROOT / "data" / "sequences" / f"{HTR1A}.fa"
    if not fasta.exists():
        pytest.skip("no CDS cache on this machine")
    seq = "".join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">"))
    cells = bf.values_in(drawio)
    assert seq[:12] in cells["484"] and seq[-12:] in cells["484"] and f"{len(seq):,} bp" in cells["484"]
    assert f"[{seq[:8]}…{seq[-8:]}]" in cells["488"]


def test_the_worked_example_window_is_dnabert2s(drawio):
    """Panel B's window and overlap are DNABERT-2's extraction settings, W in content tokens."""
    spec, cells = ENCODER_SPECS["dnabert2"], bf.values_in(drawio)
    for cell in ("460", "487"):
        assert f"W = {spec.max_content_tokens}" in _text(cells[cell]), cell
    assert f"overlap by {spec.stride} tokens" in _text(cells["489"])


def test_panel_a_prints_each_encoders_chunk_length(drawio):
    """Panel A's W column is the CDS chunk length in content tokens, the appendix's numbers,
    not the model context window (NT-v2 2,048 and HyenaDNA 1M were printed before Oct 4)."""
    cells = bf.values_in(drawio)
    assert _text(cells["616"]) == "Chunk W (tokens)"
    for name_cell, w_cell, enc in (("618", "619", "dnabert2"), ("621", "622", "nt_v2"),
                                   ("624", "625", "gena_lm"), ("627", "628", "hyena_dna")):
        assert ENCODER_SPECS[enc].display_name.startswith(_text(cells[name_cell]).strip()), name_cell
        assert _text(cells[w_cell]) == f"{ENCODER_SPECS[enc].max_content_tokens:,}", enc


def test_a_moved_chunk_length_is_caught(numbers, drawio, monkeypatch):
    moved = dict(ENCODER_SPECS, nt_v2=dataclasses.replace(ENCODER_SPECS["nt_v2"], max_content_tokens=999))
    monkeypatch.setattr(bf, "ENCODER_SPECS", moved)
    assert [cell for cell, _, _ in bf.stale_cells(drawio, numbers)] == ["622"]


def test_the_worked_example_is_htr1as_one_chunk_and_its_embedding(drawio):
    """Panel B draws as many chunks as the v2 cache holds for HTR1A (one), and prints that
    chunk's token mean (first two dims) as e_c1."""
    spec = ENCODER_SPECS["dnabert2"]
    cache = spec.chunk_dir / f"{HTR1A}.npz"
    if not cache.exists():
        pytest.skip("no v2 DNABERT-2 chunk cache on this machine")
    with np.load(cache, allow_pickle=False) as z:
        meta, mean = json.loads(str(z["meta"])), z["mean"]
    assert (meta["model"], meta["revision"], meta["max_content_tokens"], meta["stride"]) == \
        (spec.model_name, spec.revision, spec.max_content_tokens, spec.stride)
    cells = bf.values_in(drawio)
    boxes = [c for c, value in cells.items() if re.match(r"c\d+\s", _text(value))]
    assert len(boxes) == mean.shape[0] == 1, boxes
    printed = [float(x.replace("\u2212", "-")) for x in re.findall(r"\u2212?\d+\.\d+", cells[bf.EXAMPLE_EMBEDDING])]
    assert printed == [round(float(v), 3) for v in mean[0, :2]]


def test_a_moved_number_is_caught(numbers, drawio):
    moved = {**numbers, "cds.f5.aa-kmer": "0.999"}
    assert [cell for cell, _, _ in bf.stale_cells(drawio, moved)] == ["538"]


def test_filling_rewrites_only_the_builders_cells(numbers, drawio):
    moved = {**numbers, "cds.gp.nt-kmer": "0.111", "n.family.ion": "999"}
    before, after = bf.values_in(drawio), bf.values_in(bf.fill(drawio, bf.cell_values(moved)))
    assert {c for c in before if before[c] != after[c]} == {"664", "442"}
    assert set(before) == set(after)
    assert bf.fill(drawio, bf.cell_values(numbers)) == drawio        # idempotent on the committed file


def test_the_stamp_matches_the_drawio_and_the_png():
    stamp = json.loads(bf.STAMP.read_text())
    assert bf.stamp_problems(bf.DRAWIO.read_bytes(), bf.PNG.read_bytes(), stamp) == []


def test_an_edit_without_a_render_is_caught():
    stamp = json.loads(bf.STAMP.read_text())
    drawio, png = bf.DRAWIO.read_bytes(), bf.PNG.read_bytes()
    assert len(bf.stamp_problems(drawio + b" ", png, stamp)) == 1
    assert len(bf.stamp_problems(drawio, png + b"\0", stamp)) == 1


def test_no_result_decimal_outside_the_builders_cells(drawio):
    assert bf.stray_decimals(drawio) == []
    typed = re.sub(r'(<mxCell id="537" value=")[^"]*', r"\1AA 2-mer: 0.735", drawio)
    assert bf.stray_decimals(typed) == [("537", "0.735")]


def test_tex_values_become_figure_text_or_are_refused():
    assert bf.plain(r"3{,}244") == "3,244"
    assert bf.plain(r"\ensuremath{-}0.021") == "−0.021"
    with pytest.raises(ValueError, match="dagger"):
        bf.plain(r"0.580\sens{}")
    with pytest.raises(ValueError, match="TeX"):
        bf.plain(r"0.5\,M")
