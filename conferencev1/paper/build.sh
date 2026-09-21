#!/usr/bin/env bash
# Compile the paper: pdflatex -> bibtex -> pdflatex x2. Run from paper/ dir.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
