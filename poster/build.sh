#!/usr/bin/env bash
# Build the MINA poster PDF (41in x 36in). Run from the poster/ directory.
# pdflatex twice so tcolorbox/tikz positions settle.
set -euo pipefail
cd "$(dirname "$0")"

pdflatex --interaction=nonstopmode --halt-on-error mina_poster.tex >/dev/null
pdflatex --interaction=nonstopmode --halt-on-error mina_poster.tex

echo "--- page size ---"
pdfinfo mina_poster.pdf | grep -E "Page size|Page rot"
echo "--- fonts (expect all emb=yes) ---"
pdffonts mina_poster.pdf
