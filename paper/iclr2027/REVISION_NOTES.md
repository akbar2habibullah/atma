# ICLR 2027 revision notes

## Current milestone (2026-09-05)

A manuscript-only evidence and presentation revision is complete. No new
training or benchmark runs were performed. The ICLR and arXiv sources and PDFs
are synchronized, preserving their anonymous and public wrappers respectively.

The revision:

- Centers the abstract, introduction, and conclusion on attention–memory
  interactions and checkpoint-specific retention failure.
- Adds a main-text component discussion and a table placing both fresh paired
  runs beside the primary NoPE/Polar checkpoints. The smaller fresh Polar exact
  retrieval results and the microbatch confound are explicit.
- Separates untouched architectural comparisons from the retention intervention.
  The primary length figure and downstream bars show untouched checkpoints; the
  full intervention curves remain in Appendix J and the main audit keeps its
  before/after endpoint table.
- Replaces the crowded main haystack matrix with fresh-run endpoints and keeps
  the complete haystack breakdowns in the appendix.
- Regenerates untouched retrieval tables from full runs, excluding smoke tests.
  For example, Atma-Raven-Titans synthetic exact retrieval at 2K is 8.0%, not
  the smoke-contaminated 20.7%.
- Corrects the BABILong captions: untouched NoPE is 39/21/6% at 8/16/32K and
  zero from 64K onward. Clarifies that small downstream-mean changes do not
  establish preservation of every individual metric.
- States that retrieval reuses one real-text stream and that BPB scores 18 fixed
  256-token spans (4,608 target tokens per model and context length).
- Separates preallocated engine memory from occupied per-sequence KV storage;
  the 128K BF16 KV size of 0.5 GiB is an analytical calculation, not a new memory
  measurement.
- Corrects the NoPE loss-figure annotation and scopes the microbatch diagnostic.
- Restores Times-family text fonts through T1 encoding. Regenerates affected PDF
  assets and marks PDFs binary to prevent Git line-ending corruption.

The rebuilt ICLR draft has 33 pages. Main text ends on page 9; the AI use and
reproducibility statements and references begin on page 10. Appendix A starts on
page 11 after the remaining references. The fresh-run table is in the main text
on page 6, with full replication results in Appendix F. Figures, tables, and
page boundaries were visually inspected. Both builds have no unresolved
references, missing glyphs, font-substitution warnings, or overfull boxes;
non-fatal underfull spacing warnings remain.

## Evidence boundaries retained

1. The complete factorial tests interactions across recipe settings, not seeds.
2. The full Polar mechanism's incremental benefit over a temperature-matched
   softmax control remains unresolved; the fixed-null component pilot is
   reported candidly.
3. The NoPE likelihood repair repeats in both fresh pairs. The primary Polar
   cap benefit and the primary extreme-length exact-retrieval magnitude do not.
4. Raven is a separate model-family/optimizer comparison. Exact natural-text
   retrieval at extreme lengths remains unsolved, and serving timings have one
   sample per cell.
5. The retention cap diagnoses existing checkpoints; the training origin and a
   bounded train-time formulation remain open.
6. No matched 9.816B-token TDA result is included. Its potential addition before
   submission is future work, not evidence claimed by this version.

## Reproducible manuscript checks

- `python paper/generate_primary_tables.py --check` checks the generated main
  endpoint tables and appendix retrieval tables against archived full results.
- `python paper/generate_re_evaluation_tables.py` rebuilds diagnostic tables in
  both editions, including corrected captions.
- `python paper/sync_arxiv.py` carries shared prose and assets into the public
  wrapper. See `paper/README.md` for the complete regeneration/build sequence.
- Both PDFs were compiled with Tectonic 0.17.0. Training code and experiment logs
  are unchanged.

## Pending author-led stages

### Stage 2: responsible-content audit

- Trace every headline number to an archived JSON/log/table.
- Recheck denominators, aggregation rules, dataset names, adaptation details, and hardware/software versions.
- Audit every causal phrase; retain causal wording only where the intervention supports it.
- Verify that all related work is cited accurately and described in third person under double-blind review.
- Confirm that the anonymous artifact and manuscript do not expose author identity.
- Expand the AI Use Statement to cover the complete research history, including every ICLR-required disclosure category that applies.

### Stage 3: author rewrite

- Rewrite for authorial understanding and precision, not merely surface variation.
- Preserve equation semantics, numerical values, qualifiers, citations, labels, and comparison-group boundaries.
- Keep the main-text boundary at or before page 9 after every substantial rewrite.

### Stage 4: final sanity check

Codex should report issues only. It should not rewrite the authors' final prose. The check should cover page limit, anonymity, unresolved references, claim/evidence consistency, table-text agreement, AI disclosure completeness, typography, and PDF rendering.

## Verified ICLR 2027 constraints

- Abstract deadline: September 18, 2026 AOE.
- Full paper deadline: September 25, 2026 AOE.
- Initial main text: at most 9 pages.
- References and appendices: excluded from the main-text limit.
- AI use statement: mandatory and excluded from the main-text limit.
