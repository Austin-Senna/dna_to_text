# MINA poster — Columbia Undergraduate Research Symposium 2026

Print-ready poster for MINA (MLCB 2026 spotlight), general-audience "balanced" cut.

- **Event:** Fri Oct 23, 2026, Roone Arledge Auditorium, Lerner Hall.
- **Submission deadline:** Thu Sept 24, 11:59 PM — to BOTH Columbia Print Services
  (https://columbia.webdeskprint.com, site "Columbia Undergraduate Research Symposium",
  pickup Morningside–Journalism–Room 106) AND the URF Poster Submission Form.
- **Size:** 41 in wide × 36 in tall, landscape (hard requirement; auto-rejected otherwise).

## Build

```bash
# 1. Figures (vector PDF, reuses the paper's metric accessors so numbers match):
uv run poster/build_poster_figures.py        # writes poster/figures/*.pdf

# 2. Poster (pdflatex twice; prints page size + font-embedding report):
bash poster/build.sh                          # writes poster/mina_poster.pdf
```

Engine note: this TeX Live has no `luaotfload`, so lualatex+fontspec is unavailable.
The poster uses `helvet` (URW Helvetica, Type1) under **pdflatex** — embeds cleanly,
visually identical to Helvetica/TeX Gyre Heros.

## Verification (all currently passing)

```bash
pdfinfo  mina_poster.pdf | grep "Page size"   # -> 2952 x 2592 pts (= 41 x 36 in)
pdffonts mina_poster.pdf | grep -c "Type 3"   # -> 0 (all fonts embedded)
```

- Body text ≈ 33 pt (target ≥24 pt); title/subhead per URF guidance.
- White background, thin-bordered panels (respects the "no solid-colour blocks" ink rule).
- Every printed statistic traces to the paper tables (homology-aware split, not random).

## Content = paper, distilled

Story: **"Do AI models of DNA actually understand biology?"** Real family signal from
coding sequence, but only up to protein composition (NT-v2 0.727 vs AA-2mer 0.735,
ESM-2 0.960); regulatory (TSS) context collapses to near chance (0.313); GenePT text
recovery weak (R²=0.077). Numbers are the accepted MLCB results.

## Open items before printing (see also ~/.claude/plans/…)

1. **QR code** — currently a placeholder. Decide target (repo / paper PDF / Zenodo DOI),
   then generate and drop into the header box.
2. **Faculty mentor / lab** — URF template expects "Name, Mentor's Name/Lab". Confirm
   whether a mentor line is required or the affiliation stays "Columbia University, CS".
3. **Title wording** — subtitle currently reads "linear probes"; camera-ready may switch
   to "linear evaluation". Pick one.
4. **URF title/abstract** — confirm what was submitted at acceptance; edits allowed only
   until Sept 24 (email ugrad-urf@columbia.edu).
5. **Proofread** — one non-specialist read per URF guidance.

## Files

- `mina_poster.tex` — the poster (beamerposter, single file).
- `build_poster_figures.py` — poster figures; reuses `scripts/build_result_figures.py`
  accessors + `scripts/build_umap_compare.py` UMAP compute (paper scripts untouched).
- `build.sh` — pdflatex ×2 + verification prints.
- `figures/` — generated vector PDFs (regenerable; safe to delete and rebuild).
