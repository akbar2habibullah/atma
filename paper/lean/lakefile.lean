import Lake
open Lake DSL

package «atma-lean» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.30.0"

@[default_target]
lean_lib Atma where
  roots := #[`AtmaBounds, `AtmaProbability]
