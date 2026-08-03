/-!
ATMA proof skeletons checked with core Lean 4 only.

This file intentionally avoids mathlib. It verifies the deterministic proof
structure used in Appendix A of the paper:

* finite survivor/odds sums are bounded by `n * perKeyBound`;
* Polar content + count + memory is bounded when each channel is bounded;
* a memory-state ball is invariant if one update step maps the ball into itself.

The real-analysis and probability layer in the paper (sub-Gaussian tails,
exponentials, and extreme-value asymptotics) should be formalized separately with
mathlib. Here those analytic facts appear as explicit finite, natural-valued
premises.
-/

namespace AtmaBounds

def sumList : List Nat -> Nat
  | [] => 0
  | x :: xs => x + sumList xs

def allLe (xs : List Nat) (b : Nat) : Prop :=
  match xs with
  | [] => True
  | x :: rest => x <= b ∧ allLe rest b

theorem sumList_le_length_mul_bound :
    ∀ (xs : List Nat) (b : Nat), allLe xs b -> sumList xs <= xs.length * b
  | [], b, _ => by
      simp [sumList]
  | x :: xs, b, h => by
      have hx : x <= b := h.left
      have hxs : allLe xs b := h.right
      have ih : sumList xs <= xs.length * b :=
        sumList_le_length_mul_bound xs b hxs
      calc
        sumList (x :: xs) = x + sumList xs := rfl
        _ <= b + xs.length * b := Nat.add_le_add hx ih
        _ = xs.length * b + b := Nat.add_comm b (xs.length * b)
        _ = xs.length.succ * b := by
          rw [Nat.succ_mul]
        _ = (x :: xs).length * b := rfl

/- A finite analogue of the TDA/Polar survivor-count bound:
   if every per-key survivor contribution is at most `perKey`, then the total
   survivor contribution is at most `number_of_keys * perKey`. -/
theorem survivor_sum_bound (survivors : List Nat) (perKey : Nat)
    (h : allLe survivors perKey) :
    sumList survivors <= survivors.length * perKey :=
  sumList_le_length_mul_bound survivors perKey h

/- A finite analogue of the Polar null-odds bound:
   if every real-noise-to-null odds contribution is at most `perKeyOdds`, then
   the total noise odds are at most `number_of_noise_keys * perKeyOdds`. -/
theorem null_odds_sum_bound (odds : List Nat) (perKeyOdds : Nat)
    (h : allLe odds perKeyOdds) :
    sumList odds <= odds.length * perKeyOdds :=
  sumList_le_length_mul_bound odds perKeyOdds h

/- Polar's paper proof uses two bounded channels:
   content is bounded after unit-direction normalization and fixed projection,
   count is bounded because `mag` lies in `[0,1)` before fixed projection.
   This theorem checks the deterministic composition step. -/
theorem polar_channels_bounded
    (content count contentBound countBound : Nat)
    (hc : content <= contentBound)
    (hm : count <= countBound) :
    content + count <= contentBound + countBound :=
  Nat.add_le_add hc hm

/- ATMA adds the memory branch to the bounded Polar readout. -/
theorem atma_mixer_bounded
    (content count memory contentBound countBound memoryBound : Nat)
    (hc : content <= contentBound)
    (hcnt : count <= countBound)
    (hmem : memory <= memoryBound) :
    content + count + memory <= contentBound + countBound + memoryBound := by
  exact Nat.add_le_add (Nat.add_le_add hc hcnt) hmem

/- If a single memory update maps the ball `{m | m <= bound}` into itself,
   then all recurrent states remain in the ball. In the paper, the analytic
   contractive-gate conditions prove this one-step premise. -/
theorem memory_ball_invariant
    (state : Nat -> Nat)
    (bound : Nat)
    (h0 : state 0 <= bound)
    (hstep : ∀ t, state t <= bound -> state (t + 1) <= bound) :
    ∀ t, state t <= bound := by
  intro t
  induction t with
  | zero =>
      exact h0
  | succ t ih =>
      exact hstep t ih

/- Combining the invariant memory ball with bounded content/count gives a
   length-independent bound for every recurrent step. -/
theorem atma_mixer_bounded_all_steps
    (content count : Nat -> Nat)
    (memory : Nat -> Nat)
    (contentBound countBound memoryBound : Nat)
    (hc : ∀ t, content t <= contentBound)
    (hcnt : ∀ t, count t <= countBound)
    (hmem0 : memory 0 <= memoryBound)
    (hmemStep : ∀ t, memory t <= memoryBound -> memory (t + 1) <= memoryBound) :
    ∀ t, content t + count t + memory t
      <= contentBound + countBound + memoryBound := by
  intro t
  have hmem : memory t <= memoryBound :=
    memory_ball_invariant memory memoryBound hmem0 hmemStep t
  exact atma_mixer_bounded
    (content t) (count t) (memory t)
    contentBound countBound memoryBound
    (hc t) (hcnt t) hmem

end AtmaBounds
