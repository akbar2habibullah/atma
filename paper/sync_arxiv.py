"""Sync the current ICLR manuscript into the public arXiv wrapper and assets."""

from pathlib import Path
import shutil


def main():
    paper = Path(__file__).resolve().parent
    source = (paper / "iclr2027/iclr2027_conference.tex").read_text(encoding="utf-8")
    source = source.replace(r"\usepackage{iclr2027_conference,times}", r"\usepackage{atma_arxiv,times}")
    source = source.replace(r"\graphicspath{{../}}", r"\graphicspath{{./}}")
    source = source.replace(r"\author{Anonymous Authors}", r"""\author{Habibullah Akbar \\
Kreasof AI \\
Jakarta, Indonesia \\
\texttt{habibullah.akbar@kreasof.my.id}}""")
    source = source.replace(r"% \iclrfinalcopy", r"\iclrfinalcopy")
    source = source.replace(r"\end{abstract}", r"Code: \url{https://github.com/kreasof-ai/atma}" + "\n" + r"\end{abstract}")
    source = source.replace("The anonymous artifact at", "The public artifact at")
    source = source.replace("https://anonymous.4open.science/r/atma", "https://github.com/kreasof-ai/atma")
    (paper / "arxiv/atma_arxiv.tex").write_text(source, encoding="utf-8")
    for name in ("appendix.tex", "gamma_re_evaluation_tables.tex", "appendix_table.tex",
                 "colm2026_conference.bib", "iclr2027_additions.bib"):
        shutil.copyfile(paper / "iclr2027" / name, paper / "arxiv" / name)
    for path in paper.glob("fig_*.pdf"):
        shutil.copyfile(path, paper / "arxiv" / path.name)
    print("Synced manuscript, shared appendices, bibliography, and figures to paper/arxiv")


if __name__ == "__main__":
    main()
