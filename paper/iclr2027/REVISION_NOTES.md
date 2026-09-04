# ICLR 2027 revision notes

## Current milestone

Stage 1 (ICLR conversion and length reduction) is complete. The source and
compiled PDF now also contain the full retention-horizon diagnostic: parameter
inspection, the paired single-head intervention, and independent downstream,
retrieval, BPB, and BABILong re-evaluation. The appendix additionally reports
the matched single-seed Polar component pilot and scopes its unexpected
fixed-null result as an unresolved calibration question. It now also reports two
seed-paired 9.816B-token NoPE/Polar replications, restricted as prespecified to BPB
and retrieval in untouched and 256-token-capped conditions. The full re-evaluation
is presented as length-wise curves in the main paper and as generated,
cell-complete tables in the appendix. In the current compiled draft:

- The main body concludes on page 9, within the nine-page limit.
- The page-limit-exempt AI use and reproducibility statements follow on page 9.
- References begin on page 10 and appendices begin on page 11.
- The compiled artifact is 34 pages; the two replication tables appear in Appendix F and the nine full diagnostic and re-evaluation tables appear in Appendix J.

The retention-horizon figures, endpoint table, and prose therefore fit without exceeding the
main-text limit, but leave no additional full page of headroom.

## Preserved main-paper spine

1. Long-context modeling is framed as a Pareto problem across retrieval, reasoning, document likelihood, short-context quality, and systems cost.
2. The Polar mechanism retains its null floor, bounded direction/magnitude decomposition, and gated-delta memory channel.
3. The main evidence retains the recipe ablation, matched long-context endpoints, synthetic/real-text boundary, short-context and systems trade-offs, and adds a checkpoint-level retention-horizon audit.
4. Claims are explicitly scoped: Raven is not an optimizer-matched ablation, 256K natural-text retrieval remains unsolved, serving measurements are descriptive, the infrastructure transition is not causal evidence, and the inference cap is a diagnostic rather than a validated training method.
5. Both Raven Native and Atma-Raven-Titans remain visible as untouched references in the main length curves, short-context bars, retrieval depth heatmaps, serving figure, article retrieval chart, and web recovery chart. Checkpoint-specific probes that were not run on Raven say so explicitly rather than implying missing results.

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
