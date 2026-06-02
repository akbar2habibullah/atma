"""Prototype: Titans-style linear compression memory for the Polar attention layer.

This is the standalone design-validation script for the MAG memory branch (Step 1
of plans/linked-forging-sparrow.md), mirroring the role `polar_proto.py` played for
polar attention. It implements the *linear matrix* long-term memory (the gated delta
rule), proves the chunked-parallel form equals the sequential scan, and demonstrates
the load-bearing claim: the forget gate makes the memory length-invariant.

Math
----
Titans neural memory with associative loss l = ||M k - v||^2, one inner GD step
(lr theta_t), momentum eta=0, forget/decay gate alpha_t:

    M_t = (1 - alpha_t) M_{t-1} - theta_t (M_{t-1} k_t - v_t) k_t^T

Reparametrize gamma_t = 1 - alpha_t in (0,1)  (decay),
              beta_t  = theta_t / (1 - alpha_t) in (0,1)  (write strength).
Then this is exactly the **Gated DeltaNet** recurrence

    M_t = gamma_t ( M_{t-1} + w_t k_t^T ),   w_t = beta_t (v_t - M_{t-1} k_t)

with a causal readout r_t = M_{t-1} q_t (query reads memory of strictly past tokens).

Why gamma matters (the invariance gate) -- corrected empirically by the sweep
-----------------------------------------------------------------------------
Initial hypothesis was "gamma==1 -> state is a random walk -> ||M_N|| ~ sqrt(N)". The
sweep DISPROVES that for the *delta* rule: the (I - beta k k^T) key-replacement makes it
an online regression that stays norm-bounded even at gamma==1 (state converges toward
the least-squares M ~ V K^+, bounded by ~d associations). The thing that genuinely blows
up ~sqrt(N) is a *Hebbian* / linear-attention memory (S_t = gamma S_{t-1} + v k^T, no
delta correction) at gamma==1 -- there the forget gate is what bounds the state.

So the forget gate's real job in the delta memory is not norm-bounding (the delta
structure already gives that) but setting a *fixed temporal horizon* ~1/(1-gamma):
gamma<1 forgets by recency, gamma==1 forgets only by interference (capacity ~d). This is
a genuine tension for the needle test: gamma<1 keeps readout stats tightest / most
length-invariant but DECAYS a distant planted association; gamma==1 retains it until
overwritten. `invariance_sweep()` shows all three regimes.

Conventions
-----------
Tensors are batched (B, H, N, d): q,k shaped (...,dk); v shaped (...,dv); gamma,beta
shaped (B, H, N). State M shaped (B, H, dv, dk). Everything runs on CPU in fp64 by
default (this is a correctness prototype, not a kernel).
"""

import torch


# --------------------------------------------------------------------------- #
# Reference oracle: sequential token-by-token scan (the definition).
# --------------------------------------------------------------------------- #
def gated_delta_sequential(q, k, v, gamma, beta, S0=None):
    """Sequential gated delta rule. Returns (readouts R (B,H,N,dv), final state (B,H,dv,dk)).

    Causal readout: r_t uses M_{t-1} (state BEFORE writing token t)."""
    B, H, N, dk = q.shape
    dv = v.shape[-1]
    S = torch.zeros(B, H, dv, dk, dtype=q.dtype, device=q.device) if S0 is None else S0
    R = []
    for t in range(N):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]          # (B,H,dk),(B,H,dk),(B,H,dv)
        gt, bt = gamma[:, :, t], beta[:, :, t]                   # (B,H)
        r = torch.einsum("bhvd,bhd->bhv", S, qt)                 # readout (pre-write)
        R.append(r)
        pred = torch.einsum("bhvd,bhd->bhv", S, kt)              # M_{t-1} k_t
        w = bt[..., None] * (vt - pred)                          # (B,H,dv)
        S = gt[..., None, None] * (S + torch.einsum("bhv,bhd->bhvd", w, kt))
    return torch.stack(R, dim=2), S


def hebbian_sequential(q, k, v, gamma, beta, S0=None):
    """Plain gated linear-attention (Hebbian) memory: no delta correction.
    S_t = gamma_t S_{t-1} + beta_t v_t k_t^T. Used only to contrast norm growth in the
    invariance sweep -- at gamma==1 this grows ~sqrt(N) (the genuine blow-up)."""
    B, H, N, dk = q.shape
    dv = v.shape[-1]
    S = torch.zeros(B, H, dv, dk, dtype=q.dtype, device=q.device) if S0 is None else S0
    R = []
    for t in range(N):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        gt, bt = gamma[:, :, t], beta[:, :, t]
        R.append(torch.einsum("bhvd,bhd->bhv", S, qt))
        S = gt[..., None, None] * S + bt[..., None, None] * torch.einsum("bhv,bhd->bhvd", vt, kt)
    return torch.stack(R, dim=2), S


# --------------------------------------------------------------------------- #
# Chunked-parallel form (UT transform) lives in model/blocks.py (single source of
# truth, used by the model). Re-imported here so this prototype's sequential oracle
# and invariance sweep validate the *production* code path via verify_titans.py.
# --------------------------------------------------------------------------- #
from model.blocks import gated_delta_chunked  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Helpers: synthetic inputs with UNIT-norm q/k (required for delta-rule stability).
# --------------------------------------------------------------------------- #
def make_inputs(B, H, N, dk, dv, gamma_mean=0.98, beta_mean=0.5, gamma1=False, seed=0,
                dtype=torch.float64, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H, N, dk, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, H, N, dk, generator=g, dtype=dtype, device=device)
    v = torch.randn(B, H, N, dv, generator=g, dtype=dtype, device=device)
    # L2-normalize q,k to UNIT norm. The delta rule's per-step operator eigenvalue in the
    # key direction is gamma*(1 - beta*||k||^2); ||k||=1 keeps it in (0,1) (stable). This is
    # distinct from polar's RMS-norm (which gives ||k||^2 = dk and makes the rule expansive).
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if gamma1:
        gamma = torch.ones(B, H, N, dtype=dtype, device=device)
    else:
        # data-dependent gate centered on gamma_mean (logit + small noise)
        bias = torch.log(torch.tensor(gamma_mean / (1 - gamma_mean), dtype=dtype))
        gamma = torch.sigmoid(bias + 0.3 * torch.randn(B, H, N, generator=g, dtype=dtype, device=device))
    bbias = torch.log(torch.tensor(beta_mean / (1 - beta_mean), dtype=dtype))
    beta = torch.sigmoid(bbias + 0.3 * torch.randn(B, H, N, generator=g, dtype=dtype, device=device))
    return q, k, v, gamma, beta


# --------------------------------------------------------------------------- #
# Invariance sweep: the load-bearing claim (forget gate => length-invariant memory).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def invariance_sweep(Ns=(256, 512, 1024, 2048, 4096, 8192, 16384), dk=32, dv=32):
    print("\n=== Invariance sweep: state-norm ||S_N||_F vs N (mean over heads) ===")
    print(f"{'N':>7} | {'delta g=.98':>12} {'delta g=1':>12} | {'hebb g=.98':>12} {'hebb g=1':>12}")
    print("-" * 64)
    for N in Ns:
        qd, kd, vd, gd, bd = make_inputs(1, 4, N, dk, dv, gamma_mean=0.98, gamma1=False, seed=1)
        q1, k1, v1, g1, b1 = make_inputs(1, 4, N, dk, dv, gamma1=True, seed=1)
        sn_d98 = gated_delta_sequential(qd, kd, vd, gd, bd)[1].norm(dim=(-2, -1)).mean().item()
        sn_d1 = gated_delta_sequential(q1, k1, v1, g1, b1)[1].norm(dim=(-2, -1)).mean().item()
        sn_h98 = hebbian_sequential(qd, kd, vd, gd, bd)[1].norm(dim=(-2, -1)).mean().item()
        sn_h1 = hebbian_sequential(q1, k1, v1, g1, b1)[1].norm(dim=(-2, -1)).mean().item()
        print(f"{N:>7} | {sn_d98:>12.3f} {sn_d1:>12.3f} | {sn_h98:>12.3f} {sn_h1:>12.3f}")
    print("\nReading: delta columns FLAT in N (self-stabilizing via key replacement, both gammas).")
    print("         hebb g=1 GROWS ~sqrt(N) (true blow-up); hebb g=.98 flat (gate bounds it).")
    print("         => delta memory is length-invariant in norm for free; the forget gate")
    print("            sets the temporal HORIZON (recall-vs-perplexity knob), not the norm.")


def _parity_demo():
    print("=== Forward parity: chunked vs sequential (fp64) ===")
    q, k, v, g, b = make_inputs(2, 3, 200, 16, 16, seed=7)
    for C in (1, 16, 64, 200):
        Rseq, Sseq = gated_delta_sequential(q, k, v, g, b)
        Rchk, Schk = gated_delta_chunked(q, k, v, g, b, chunk=C)
        dR = (Rseq - Rchk).abs().max().item()
        dS = (Sseq - Schk).abs().max().item()
        print(f"  chunk={C:>3}:  max|dR|={dR:.2e}   max|dS|={dS:.2e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    _parity_demo()
    invariance_sweep()
