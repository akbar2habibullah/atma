# Paper artifacts

The current manuscript is the ICLR 2027 draft:

- source: [`iclr2027/iclr2027_conference.tex`](iclr2027/iclr2027_conference.tex)
- appendix: [`iclr2027/appendix.tex`](iclr2027/appendix.tex)
- compiled draft: [`iclr2027/iclr2027_conference.pdf`](iclr2027/iclr2027_conference.pdf)
- revision state: [`iclr2027/REVISION_NOTES.md`](iclr2027/REVISION_NOTES.md)

The figure-generation scripts in this directory read committed payloads under [`benchmarks/logs/`](../benchmarks/logs/) and write the PDF figures alongside the legacy CoLM source. The ICLR source currently references those shared figures and the shared bibliography one directory above it.

The `colm2026_*` files are the previous manuscript snapshot. They are retained for provenance and should not be used for current claims. Machine-checked proof skeletons live under [`lean/`](lean/).

## Build

Run LaTeX from `paper/` so the ICLR draft can resolve its shared figures and bibliography:

```bash
cd paper
pdflatex iclr2027/iclr2027_conference.tex
bibtex iclr2027/iclr2027_conference
pdflatex iclr2027/iclr2027_conference.tex
pdflatex iclr2027/iclr2027_conference.tex
```

Generated LaTeX auxiliaries are ignored. Commit the compiled PDF only when it represents the current reviewed source.
