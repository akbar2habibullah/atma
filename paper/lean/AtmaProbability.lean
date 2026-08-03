import Mathlib

/-!
Mathlib-backed real-valued bounds for the ATMA theoretical appendix.

This file formalizes the finite real-analysis layer that sits between TDA-style
sub-Gaussian tail assumptions and the Polar null-sink conclusions:

* finite sums of per-key tail probabilities;
* expected survivor-count bound from a per-key sub-Gaussian tail bound;
* union-bound composition from an assumed finite union bound;
* soft null-sink odds bound under a deterministic extreme-value margin.

The derivation of the sub-Gaussian tail inequality from an mgf assumption is not
included here; it is represented by `SubGaussianTailAssumption`. Formalizing that
step would require a deeper measure/probability development.
-/

open scoped BigOperators

namespace AtmaProbability

noncomputable section

def evTail (nu v : Real) : Real :=
  Real.exp (-(nu ^ 2 / (2 * v ^ 2)))

def SubGaussianTailAssumption {Key : Type} [Fintype Key]
    (eventProb : Key -> Real) (nu v : Real) : Prop :=
  forall k, eventProb k <= evTail nu v

theorem finite_sum_bound {Key : Type} [Fintype Key]
    (f : Key -> Real) (B : Real)
    (h : forall k, f k <= B) :
    (Finset.univ.sum f) <= (Fintype.card Key : Real) * B := by
  calc
    Finset.univ.sum f <= Finset.univ.sum (fun _ : Key => B) := by
      exact Finset.sum_le_sum (by intro k _hk; exact h k)
    _ = (Fintype.card Key : Real) * B := by
      simp

/- Expected survivor count when `eventProb k` is the probability that key `k`
   is a spurious survivor. This is the finite expectation step used by the paper:
   linearity reduces the expected count to a sum of per-key probabilities. -/
theorem expected_survivor_count_bound {Key : Type} [Fintype Key]
    (eventProb : Key -> Real) (nu v : Real)
    (htail : SubGaussianTailAssumption eventProb nu v) :
    (Finset.univ.sum eventProb)
      <= (Fintype.card Key : Real) * evTail nu v :=
  finite_sum_bound eventProb (evTail nu v) htail

/- If an external finite union bound has already shown that the probability of
   any exceedance is at most the sum of per-key exceedance probabilities, the
   same tail assumption bounds that any-exceedance probability. -/
theorem any_exceedance_probability_bound {Key : Type} [Fintype Key]
    (eventProb : Key -> Real) (anyProb nu v : Real)
    (hunion : anyProb <= Finset.univ.sum eventProb)
    (htail : SubGaussianTailAssumption eventProb nu v) :
    anyProb <= (Fintype.card Key : Real) * evTail nu v := by
  exact le_trans hunion (expected_survivor_count_bound eventProb nu v htail)

/- Per-key soft null-sink odds under a deterministic margin. If each irrelevant
   score is at least `delta` below the null floor, then each real-key softmax odds
   term is bounded by `exp(theta * (-delta))`. -/
theorem per_key_null_odds_bound
    (score nu delta theta : Real)
    (htheta : 0 <= theta)
    (hscore : score <= nu - delta) :
    Real.exp (theta * (score - nu))
      <= Real.exp (theta * (-delta)) := by
  have hdiff : score - nu <= -delta := by
    linarith
  have hmul : theta * (score - nu) <= theta * (-delta) :=
    mul_le_mul_of_nonneg_left hdiff htheta
  exact (Real.exp_le_exp).2 hmul

/- Summed null odds over a finite noise set. This is the soft analogue of the
   TDA survivor-count argument: TDA sums sparse survivors; Polar sums soft odds
   against the null sink. -/
theorem null_odds_sum_bound {Key : Type} [Fintype Key]
    (score : Key -> Real) (nu delta theta : Real)
    (htheta : 0 <= theta)
    (hmargin : forall k, score k <= nu - delta) :
    (Finset.univ.sum
        (fun k : Key => Real.exp (theta * (score k - nu))))
      <= (Fintype.card Key : Real) * Real.exp (theta * (-delta)) := by
  apply finite_sum_bound
  intro k
  exact per_key_null_odds_bound (score k) nu delta theta htheta (hmargin k)

/- The paper's high-probability extreme-value event supplies a concrete margin
   `delta = (s - c) * sqrt(log n)`. This theorem isolates the deterministic
   consequence once that event is established. -/
theorem null_odds_extreme_value_margin_bound {Key : Type} [Fintype Key]
    (score : Key -> Real) (nu theta s c sqrtLogN : Real)
    (htheta : 0 <= theta)
    (hmargin : forall k, score k <= nu - (s - c) * sqrtLogN) :
    (Finset.univ.sum
        (fun k : Key => Real.exp (theta * (score k - nu))))
      <= (Fintype.card Key : Real)
        * Real.exp (theta * (-((s - c) * sqrtLogN))) := by
  exact null_odds_sum_bound score nu ((s - c) * sqrtLogN) theta htheta hmargin

end

end AtmaProbability
