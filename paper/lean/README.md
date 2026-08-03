# ATMA Lean Checks

This directory contains Lean 4 checks for Appendix A of the ATMA paper.

Lean toolchain used:

```powershell
C:\Users\akbar\.cache\codex-tools\lean-4.30.0-windows\bin\lean.exe --version
```

Project build:

```powershell
cd D:\Workspace\kreasof-ai\github\atma\paper\lean
C:\Users\akbar\.cache\codex-tools\lean-4.30.0-windows\bin\lake.exe build
```

Files:

- `AtmaBounds.lean`: core-Lean deterministic proof skeleton using natural-valued bounds.
- `AtmaProbability.lean`: mathlib-backed real-valued finite-sum, tail-probability, union-bound, and null-odds lemmas.

Current scope:

- finite survivor/odds sum bounds;
- expected survivor-count bound from a per-key sub-Gaussian tail premise;
- any-exceedance probability bound from a finite union-bound premise;
- soft null-sink odds bound under a deterministic extreme-value margin;
- bounded Polar content/count composition;
- bounded ATMA content/count/memory composition;
- memory-ball invariance by induction from a one-step contractive premise.

Still represented as premises rather than derived internally:

- deriving the sub-Gaussian tail inequality from an mgf assumption;
- measure-theoretic random variables and expectation;
- asymptotic convergence statements.
