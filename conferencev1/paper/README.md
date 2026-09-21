# YOLOv8 Accuracy–Efficiency Study — IEEE Conference Paper

Comparison of three architectural modifications of YOLOv8n (SE channel attention,
Res2Net multi-scale enlargement, ShuffleNet lightweight reduction) against the
baseline, judged on accuracy **and** complexity across two datasets with 5-fold
cross-validation. Contribution = accuracy–efficiency trade-off, not an accuracy
breakthrough.

## Build
```bash
bash build.sh        # pdflatex → bibtex → pdflatex×2  →  main.pdf (5 pages)
```
Requires a TeX Live install with `pdflatex` and `bibtex` (both present here).

## Regenerate data & figures (optional)
```bash
python scripts/calculate_metrics.py    # ledger.json → metrics.json + tables numbers
python scripts/generate_figures.py     # metrics.json → figures/*.pdf
python scripts/validate_references.py   # bib consistency check (PASS)
```

## Layout
```
main.tex                  full manuscript (IEEEtran, conference)
references.bib            19 references
metrics.json             all numbers, derived from the CV ledger
tables/                  Table I (datasets), II (main results), III (relative)
figures/                 Fig 1 (workflow), 2 (SE module), 3 (trade-off scatter)
scripts/                 calculate_metrics / generate_figures / validate_references
build.sh                 compile script
page_check.txt           page-limit audit
reference_verification.md  reference tier audit
final_review.md          scientific audit + placeholder list + verdict
```

## Before submission
Fill the six placeholders listed in `final_review.md` (author identity, SE
insertion stage, training config, hardware, dataset provenance, funding). The
measured numbers do not depend on them.

Data source: `judol_sweep/ledger.json` (5-fold CV). No value is fabricated.
