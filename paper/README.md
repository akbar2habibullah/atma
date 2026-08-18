# Paper artifacts

The main manuscript is the ICLR 2027 draft under [`iclr2027/`](iclr2027/):

- **Main manuscript (ICLR 2027):**
  - Source: [`iclr2027/iclr2027_conference.tex`](iclr2027/iclr2027_conference.tex)
  - Appendix: [`iclr2027/appendix.tex`](iclr2027/appendix.tex)
  - Tables: [`iclr2027/gamma_re_evaluation_tables.tex`](iclr2027/gamma_re_evaluation_tables.tex)
  - Compiled draft: [`iclr2027/iclr2027_conference.pdf`](iclr2027/iclr2027_conference.pdf)
  - Revision notes: [`iclr2027/REVISION_NOTES.md`](iclr2027/REVISION_NOTES.md)

- **arXiv version:**
  - Source: [`arxiv/atma_arxiv.tex`](arxiv/atma_arxiv.tex)
  - Compiled draft: [`arxiv/atma_arxiv.pdf`](arxiv/atma_arxiv.pdf)

- **Archived versions:**
  - The previous CoLM 2026 snapshot has been archived under [`archive/colm2026/`](../archive/colm2026/).
  - Machine-checked proof skeletons live under [`lean/`](lean/).

## Build

### Main Paper (ICLR 2027)

From `paper/`:

```bash
cd paper
pdflatex iclr2027/iclr2027_conference.tex
bibtex iclr2027/iclr2027_conference
pdflatex iclr2027/iclr2027_conference.tex
pdflatex iclr2027/iclr2027_conference.tex
```

Or using Tectonic:

```bash
tectonic --outdir iclr2027 iclr2027/iclr2027_conference.tex
```

### arXiv Paper

From `paper/arxiv/`:

```bash
cd paper/arxiv
pdflatex atma_arxiv.tex
bibtex atma_arxiv
pdflatex atma_arxiv.tex
pdflatex atma_arxiv.tex
```

Or using Tectonic:

```bash
tectonic atma_arxiv.tex
```
