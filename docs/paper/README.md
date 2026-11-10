# Progress report

LaTeX source of the progress report: abstract, introduction and literature review in full; methodology, results and conclusion summarized in two or three sentences each. The compiled PDF is `progress_report.pdf`. The report carries no author name or affiliation.

## Build

```bash
cd docs/paper
latexmk -pdf -outdir=/tmp/progress_report progress_report.tex
cp /tmp/progress_report/progress_report.pdf .
```

pdfLaTeX + BibTeX (`natbib`, `plainnat`). Copying `progress_report.tex` and `references.bib` into Overleaf also works.

## Sources

- **References:** only the papers whose PDFs are in `references/`.
- **Baseline numbers** (Results section and abstract): `results/tables/baseline_silent_final.json`.
