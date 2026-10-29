#!/usr/bin/env bash
# Regenerate the Phase 1 figures from results/ and build docs/report/phase1.pdf.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
$PY -m analysis.figures --faults results/team-faults/comparison.csv \
    --verify-faults results/team-verify-faults/comparison.csv --out docs/report/figures
$PY -m analysis.diversity --grid results/team-faults/comparison.csv \
    --repeats results/team-faults-s43/comparison.csv results/team-faults-s44/comparison.csv \
    --pilot results/fuzz-pilot --out results/diversity.md --figure docs/report/figures/diversity.pdf
cd docs/report
for _ in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error phase1.tex > /dev/null; done
rm -f phase1.aux phase1.log phase1.out
echo "docs/report/phase1.pdf"
