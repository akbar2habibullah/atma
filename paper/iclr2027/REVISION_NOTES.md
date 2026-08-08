# ICLR 2027 revision notes

## Current milestone

Stage 1 (ICLR conversion and length reduction) is complete. In the compiled draft:

- Pages 1--7 contain the abstract, main text, tables, and main-text figures.
- Page 8 starts the page-limit-exempt AI use and reproducibility statements, followed by references.
- Appendices follow the bibliography.

This leaves two main-text pages of headroom for the authors' manual rewrite.

## Preserved main-paper spine

1. Long-context modeling is framed as a Pareto problem across retrieval, reasoning, document likelihood, short-context quality, and systems cost.
2. The Polar mechanism retains its null floor, bounded direction/magnitude decomposition, and gated-delta memory channel.
3. The main evidence retains the recipe ablation, matched long-context endpoints, synthetic/real-text boundary, short-context and systems trade-offs, and environment diagnostics.
4. Claims are explicitly scoped: Raven is not an optimizer-matched ablation, 256K natural-text retrieval remains unsolved, serving measurements are descriptive, and the hardware evidence establishes environment sensitivity rather than a unique CUDA-level cause.

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
