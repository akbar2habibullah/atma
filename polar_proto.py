"""
polar_proto.py — formula-refinement sandbox for Polar Attention.

NOT wired into the model. This isolates the polar *reduction* (direction +
saturating magnitude) and probes the four properties the design depends on:

  P1  Size-invariant direction  — c(query) ~ const as #relevant keys & total
                                   length vary (the conv-like "what").
  P2  Bounded, ordered magnitude — "few vs many" survives, value stays in (0,1)
                                   at any length (the fix vs raw log(m)).
  P3  Length-invariant counting  — with a calibrated noise floor theta, the
                                   count reflects #signal keys, not #noise keys.
  P4  OOD-length safety          — W_mu's input range at infer-length == train.

Run:  python polar_proto.py
"""

import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
NEG_INF = float("-inf")


# =====================================================================
# Polar reduction core (pure math; one head; one or many queries)
# =====================================================================
def polar_core(sigma, v, n_keys, *, null_logit, v_null, theta,
               len_gain=0.0, mag_beta=0.2, eps=1e-6):
    """
    sigma : (Tq, Tk)  raw scores; masked entries must be -inf.
    v     : (Tk, dk)  value directions.
    n_keys: (Tq,)     #valid keys per query (drives length temperature).
    Returns:
      c     : (Tq, dk) unit direction  ("what")
      mag   : (Tq,)    bounded count    ("how much"), in [0, 1)
      w_null: (Tq,)    mass drained to the null sink
      m     : (Tq,)    raw soft-count (pre-saturation), for diagnostics
    """
    Tq, Tk = sigma.shape
    dk = v.shape[-1]

    # Length temperature (Scalable-Softmax style): sharpen as context grows.
    temp = 1.0 + len_gain * torch.log(n_keys.clamp(min=1.0)).unsqueeze(-1)  # (Tq,1)

    # --- Direction channel: bounded softmax with a temp-scaled null sink ---
    null_col = torch.full((Tq, 1), float(null_logit), device=sigma.device, dtype=sigma.dtype)
    logits = torch.cat([sigma, null_col], dim=-1) * temp  # scale null too, so the
    w = torch.softmax(logits, dim=-1)                     # sink stays alive at long ctx
    w_keys, w_null = w[..., :-1], w[..., -1:]
    s = w_keys @ v + w_null * v_null                      # (Tq, dk)
    c = F.normalize(s, p=2, dim=-1, eps=eps)

    # --- Magnitude channel: saturating function of a soft count ---
    m = torch.sigmoid(sigma - theta).sum(dim=-1)          # masked -inf -> 0
    mag = torch.tanh(mag_beta * torch.log1p(m))           # in [0,1), bounded, ordered

    return c, mag, w_null.squeeze(-1), m


# ---------------------------------------------------------------------
# FINAL validated reduction (this is the spec to integrate).
# Direction + count both come from ONE temp-sharpened softmax with an
# EV-corrected null sink. No separate theta. Count = participation-ratio
# multiplicity gated by null-sink confidence; magnitude is bounded.
# ---------------------------------------------------------------------
def polar_final(sigma, v, n_keys, *, null_base, null_slope, v_null,
                g=0.30, beta=0.20, eps=1e-6):
    Tq, Tk = sigma.shape
    n = n_keys.clamp(min=1.0).unsqueeze(-1)             # (Tq,1)
    temp = 1.0 + g * torch.log(n)
    null = (null_base + null_slope * torch.sqrt(torch.log(n + 1.0)))  # (Tq,1)
    logits = torch.cat([sigma, null], dim=-1) * temp
    w = torch.softmax(logits, dim=-1)
    w_null, w_r = w[..., -1:], w[..., :-1]
    # direction ("what"): unit vector, count-blind
    s = w_r @ v + w_null * v_null
    c = F.normalize(s, p=2, dim=-1, eps=eps)
    # count ("how much"): length-invariant multiplicity, gated by confidence, bounded
    w_hat = w_r / w_r.sum(-1, keepdim=True).clamp_min(eps)
    n_eff = 1.0 / w_hat.square().sum(-1)
    m_eff = n_eff * (1.0 - w_null.squeeze(-1))
    mag = torch.tanh(beta * torch.log1p(m_eff))
    return c, mag, w_null.squeeze(-1)


def probe_final():
    hr("FINAL reduction end-to-end: same population at 1x vs 64x length")
    dk = 16
    d = F.normalize(torch.randn(dk), dim=0)
    v_null = torch.zeros(dk)
    print(f"{'length':>10} {'n_sig':>6} {'cos(c,d)':>9} {'mag':>6} {'w_null':>8}")
    for n_sig in (3, 50):
        for N in (64, 4096):
            v = torch.cat([d.expand(n_sig, dk), F.normalize(torch.randn(N, dk), dim=-1)])
            sigma = torch.cat([torch.full((n_sig,), 6.0), torch.randn(N)]).unsqueeze(0)
            c, mag, w_null = polar_final(sigma, v, torch.tensor([float(n_sig + N)]),
                                         null_base=2.0, null_slope=1.0, v_null=v_null)
            cos = F.cosine_similarity(c[0], d, dim=0).item()
            print(f"{n_sig+N:>10} {n_sig:>6} {cos:>9.4f} {mag.item():>6.3f} {w_null.item():>8.4f}")
    print("want: cos~1 and mag the same at 1x and 64x -> 'what' and 'how much' both invariant.")


def count_only(n_rel, n_noise, sig_rel, sig_noise, theta, mag_beta):
    """Closed-form count channel for extreme scales without materializing T*T."""
    m = n_rel * torch.sigmoid(torch.tensor(sig_rel - theta)) \
        + n_noise * torch.sigmoid(torch.tensor(sig_noise - theta))
    mag = torch.tanh(mag_beta * torch.log1p(m))
    return m.item(), mag.item()


def build_population(n_rel, n_noise, dk, sig_rel=8.0, sig_noise=0.0):
    """One query over n_rel aligned keys (all share direction d) + n_noise random keys."""
    d = F.normalize(torch.randn(dk), dim=0)
    v_rel = d.unsqueeze(0).expand(n_rel, dk)
    v_noise = F.normalize(torch.randn(max(n_noise, 0), dk), dim=-1) if n_noise > 0 else torch.empty(0, dk)
    v = torch.cat([v_rel, v_noise], dim=0)
    sigma = torch.cat([
        torch.full((n_rel,), sig_rel),
        torch.full((max(n_noise, 0),), sig_noise),
    ]).unsqueeze(0)                                       # (1, Tk)
    n_keys = torch.tensor([float(n_rel + max(n_noise, 0))])
    return sigma, v, n_keys, d


# default structural params (stand-ins for what training would learn)
P = dict(null_logit=0.0, v_null=torch.zeros(16), theta=6.0, len_gain=0.30, mag_beta=0.20)


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# =====================================================================
# P1 — Size-invariant direction
# =====================================================================
def probe_direction_invariance():
    hr("P1  Size-invariant direction: cos(c, d) across #relevant and length")
    dk = 16
    print(f"{'n_rel':>8} {'n_noise':>10} {'T':>10} {'cos(c,d)':>10} {'w_null':>9}")
    for n_rel in (1, 8, 256):
        for n_noise in (0, 1024, 16384):
            sigma, v, n_keys, d = build_population(n_rel, n_noise, dk)
            v_null = torch.zeros(dk)
            c, mag, w_null, m = polar_core(sigma, v, n_keys, **{**P, "v_null": v_null})
            cos = F.cosine_similarity(c[0], d, dim=0).item()
            print(f"{n_rel:>8} {n_noise:>10} {n_rel+n_noise:>10} {cos:>10.4f} {w_null[0].item():>9.4f}")
    print("expect: cos ~ 1 and ~constant -> direction is the 'what', independent of grid size.")


# =====================================================================
# P2 — Bounded, ordered magnitude  (the fix vs raw log)
# =====================================================================
def probe_magnitude_saturation():
    hr("P2  Magnitude: saturating tanh(beta*log1p(m)) vs UNBOUNDED raw log(m)")
    theta, beta = P["theta"], P["mag_beta"]
    print(f"{'n_rel':>10} {'soft_count m':>14} {'mag (bounded)':>15} {'raw log(m) [bad]':>18}")
    for n_rel in (1, 2, 10, 100, 1000, 10_000, 1_000_000):
        m, mag = count_only(n_rel, 0, 8.0, 0.0, theta, beta)
        raw_log = math.log(m + 1e-6)
        print(f"{n_rel:>10} {m:>14.3f} {mag:>15.4f} {raw_log:>18.3f}")
    print("expect: mag monotone & saturating in (0,1)  (your '0.3/0.9/0.99' spec);")
    print("        raw log(m) keeps climbing -> that is the OOD input W_mu must NOT see.")


# =====================================================================
# P3 — Length-invariant counting under noise (why theta matters)
# =====================================================================
def probe_noise_floor():
    hr("P3  Counting under noise: 50 signal keys buried in growing noise")
    beta = P["mag_beta"]
    print("signal: n_rel=50 @ sigma=8.   noise @ sigma=0.   varying theta (noise floor).")
    print(f"{'theta':>7} | " + " ".join(f"{('noise='+str(n)):>16}" for n in (0, 100, 10_000, 1_000_000)))
    for theta in (2.0, 4.0, 6.0, 8.0):
        cells = []
        for n_noise in (0, 100, 10_000, 1_000_000):
            m, mag = count_only(50, n_noise, 8.0, 0.0, theta, beta)
            cells.append(f"m={m:8.1f} mag={mag:4.2f}")
        print(f"{theta:>7.1f} | " + " ".join(f"{c:>16}" for c in cells))
    print("read: NO fixed theta is fully length-invariant. A tiny per-key leak")
    print("      sigmoid(sigma_noise-theta) * N still accumulates (the O(N) dilution we")
    print("      tried to escape, back in the COUNT channel). See length-ceiling probe.")


# =====================================================================
# Length ceiling: how far the count channel extrapolates, vs QK margin
# =====================================================================
def probe_length_ceiling():
    hr("Length ceiling: N_noise the count tolerates grows ~ e^(Delta/2) (QK margin)")
    beta, n_sig = P["mag_beta"], 50
    print(f"{'Delta':>6} {'theta(mid)':>11} {'signal mass':>12} {'leak/key':>11} "
          f"{'N_max (10% drift)':>18} {'0.1*n_sig*e^(D/2)':>18}")
    for Delta in (4.0, 6.0, 8.0, 10.0, 12.0):
        sig_rel, sig_noise = Delta, 0.0
        theta = Delta / 2                                   # place floor midway
        sig_mass = n_sig * torch.sigmoid(torch.tensor(sig_rel - theta)).item()
        leak = torch.sigmoid(torch.tensor(sig_noise - theta)).item()
        n_max = 0.1 * sig_mass / max(leak, 1e-30)           # N where noise = 10% of signal
        print(f"{Delta:>6.1f} {theta:>11.1f} {sig_mass:>12.2f} {leak:>11.2e} "
              f"{n_max:>18.3e} {0.1*n_sig*math.exp(Delta/2):>18.3e}")
    print("read: tolerated noise ~ n_signal * e^(Delta/2). QK separation Delta sets the")
    print("      raw-count budget EXPONENTIALLY, but Delta is capped by ~sqrt(head_dim)")
    print("      (RMS-normed q.k), so raw-count length tops out ~1e3. The saturating mag")
    print("      damps the rest; a relative floor would be the v2 to lift it further.")


# =====================================================================
# Count-mechanism shootout: which estimator is length-invariant?
# =====================================================================
def _abs_count(sig, theta):
    return torch.sigmoid(sig - theta).sum()

def _relfloor_count(sig, kappa=3.0, tau=0.5):
    """Relative floor: theta = mean + kappa*std over the row's scores."""
    theta = sig.mean() + kappa * sig.std()
    return torch.sigmoid((sig - theta) / tau).sum()

def _participation_count(sig, g=0.30):
    """Effective #keys from the length-temp-sharpened softmax weights: 1/sum(w^2)."""
    temp = 1.0 + g * math.log(sig.numel())
    w = torch.softmax(temp * sig, dim=-1)
    return 1.0 / w.square().sum()

def probe_count_mechanisms():
    hr("Count shootout: 50 signal keys (sigma=6) + Gaussian noise, N_noise growing")
    n_sig, Delta = 50, 6.0
    Ns = (64, 256, 1024, 4096, 16384, 65536)
    rows = {"absolute theta": [], "relative mean+3std": [], "participation 1/sum(w^2)": []}
    for N in Ns:
        sig = torch.cat([torch.full((n_sig,), Delta), torch.randn(N)])
        rows["absolute theta"].append(_abs_count(sig, Delta / 2).item())
        rows["relative mean+3std"].append(_relfloor_count(sig).item())
        rows["participation 1/sum(w^2)"].append(_participation_count(sig).item())
    print(f"{'mechanism':>26} | " + " ".join(f"{('N='+str(n)):>9}" for n in Ns) + " | drift")
    for name, vals in rows.items():
        drift = max(vals) / max(min(vals), 1e-9)
        print(f"{name:>26} | " + " ".join(f"{v:>9.1f}" for v in vals) + f" | {drift:>5.1f}x")
    print("read: true count = 50. absolute & relative-floor both drift up with N (O(N) leak");
    print("      survives a fixed floor). participation ratio stays ~50: the length")
    print("      temperature suppresses noise weights faster than N grows (needs g*Delta>=1).")
    # critical: pure noise must read as FEW effective keys, not many
    pure = [(_participation_count(torch.randn(N)).item()) for N in Ns]
    print("  pure-noise (no signal) participation count: "
          + " ".join(f"{v:.1f}" for v in pure)
          + "  <- must stay small & bounded, not ~N")


# =====================================================================
# Count v2: participation ratio gated by the null sink (confidence)
# =====================================================================
def _count_v2(sig, null_base, null_slope, g=0.30):
    """m_eff = (effective #real-key matches) * (mass NOT drained to null sink).
    null floor = null_base + null_slope*sqrt(log N)  -> tracks the noise extreme."""
    N = sig.numel()
    temp = 1.0 + g * math.log(N + 1)
    null = null_base + null_slope * math.sqrt(math.log(N + 1))
    logits = torch.cat([sig, torch.tensor([float(null)])]) * temp
    w = torch.softmax(logits, dim=-1)
    w_null, w_r = w[-1], w[:-1]
    w_hat = w_r / w_r.sum().clamp_min(1e-9)              # renorm over real keys
    n_eff = 1.0 / w_hat.square().sum()                  # multiplicity (length-invariant)
    m_eff = n_eff * (1.0 - w_null)                      # gate by confidence
    return n_eff.item(), w_null.item(), m_eff.item()

def probe_count_v2():
    hr("Count v2: PR * (1 - w_null), EV-corrected null floor = base + slope*sqrt(logN)")
    beta = P["mag_beta"]
    Ns = (64, 1024, 16384, 262144)
    for tag, (base, slope) in (("FIXED null (base=3, slope=0)", (3.0, 0.0)),
                               ("EV   null (base=2, slope=1)", (2.0, 1.0))):
        print(f"\n  {tag}")
        print(f"{'case':>22} | " + " ".join(f"{('N='+str(n)):>18}" for n in Ns))
        for label, n_sig in (("50 signal + noise", 50), ("PURE noise", 0)):
            cells = []
            for N in Ns:
                sig = torch.cat([torch.full((n_sig,), 6.0), torch.randn(N)]) if n_sig else torch.randn(N)
                n_eff, w_null, m_eff = _count_v2(sig, base, slope)
                mag = math.tanh(beta * math.log1p(m_eff))
                cells.append(f"m={m_eff:6.1f} mag={mag:.2f}")
            print(f"{label:>22} | " + " ".join(f"{c:>18}" for c in cells))
    print("\nwant: signal -> m~50 flat; pure-noise -> m~0. Fixed null is overtaken by the")
    print("      noise max (~sqrt(2 logN)); the sqrt(logN) floor tracks it. Distractor loss")
    print("      learns (base, slope); the SAME softmax weights feed direction + count.")


# =====================================================================
# P4 — OOD-length safety for W_mu
# =====================================================================
def probe_ood_length():
    hr("P4  OOD-length safety: input range to W_mu at train vs 32x infer length")
    beta, theta = P["mag_beta"], 8.0
    # realistic operating set: 1..200 signal keys, noise rejected by theta
    def range_over(T_noise):
        vals = [count_only(n, T_noise, 8.0, 0.0, theta, beta)[1] for n in (1, 5, 50, 200)]
        raw = [math.log(count_only(n, T_noise, 8.0, 0.0, theta, beta)[0] + 1e-6) for n in (1, 5, 50, 200)]
        return (min(vals), max(vals)), (min(raw), max(raw))
    for label, T in (("train  T=128", 128), ("infer  T=4096 (32x)", 4096)):
        (mlo, mhi), (rlo, rhi) = range_over(T)
        print(f"{label:>22}:  bounded mag in [{mlo:.3f}, {mhi:.3f}]   "
              f"|  raw log(m) in [{rlo:.2f}, {rhi:.2f}]")
    print("expect: bounded-mag range is ~identical train vs infer (W_mu stays in-distribution);")
    print("        raw-log range shifts with length (W_mu extrapolates -> the failure we avoid).")


# =====================================================================
# Distractor calibration: theta is actually learnable from noise
# =====================================================================
def probe_distractor_calibration():
    hr("Distractor loss: needs a signal-preservation counterweight or theta runs away")
    dk = 16

    def scores(q, k):                                   # realistic: QK-norm then dot/sqrt(dk)
        q = F.rms_norm(q, (dk,)); k = F.rms_norm(k, (dk,))
        return (q @ k.T) * (dk ** -0.5)

    q = torch.randn(1, dk)
    k_sig = q + 0.05 * torch.randn(1, dk)               # an aligned signal key
    sig_score = scores(q, k_sig).item()

    def report(tag, theta):
        kn = torch.randn(4096, dk)
        noise = torch.sigmoid(scores(q, kn).squeeze(0) - theta).mean().item()
        keep = torch.sigmoid(torch.tensor(sig_score) - theta).item()
        print(f"{tag:>34}: theta={theta.item():+.2f}  noise/key={noise:.4f}  signal kept={keep:.3f}")

    print(f"signal score (q.k_sig)/sqrt(dk) = {sig_score:.2f}")

    # (a) noise-rejection ONLY  ->  theta -> +inf, collapses the whole channel
    th_a = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([th_a], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.sigmoid(scores(q, torch.randn(4096, dk)).squeeze(0) - th_a).mean()
        loss.backward(); opt.step()
    report("reject-only (collapses)", th_a)

    # (b) reject noise + preserve signal  ->  theta settles at an equilibrium
    th_b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([th_b], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        reject = torch.sigmoid(scores(q, torch.randn(4096, dk)).squeeze(0) - th_b).mean()
        preserve = (1 - torch.sigmoid(scores(q, k_sig).squeeze(0) - th_b)).mean()
        (reject + preserve).backward(); opt.step()
    report("reject+preserve (settles)", th_b)
    print("read: distractor loss alone drives theta->inf (count channel dies). In the real")
    print("      model the task loss is the counterweight; a guard term makes it explicit.")


# =====================================================================
# Numerical stability
# =====================================================================
def probe_stability():
    hr("Stability: outputs finite & bounded at extreme scale")
    dk = 16
    sigma, v, n_keys, d = build_population(64, 4096, dk)
    c, mag, w_null, m = polar_core(sigma, v, n_keys, **{**P, "v_null": torch.zeros(dk)})
    print(f"T=4160 tensor run: |c|={c.norm(dim=-1).item():.4f} (==1), mag={mag.item():.4f}, "
          f"finite={torch.isfinite(c).all().item() and torch.isfinite(mag).all().item()}")
    m_huge, mag_huge = count_only(1_000_000_000, 0, 8.0, 0.0, P["theta"], P["mag_beta"])
    print(f"closed-form at 1e9 signal keys: m={m_huge:.3e}, mag={mag_huge:.6f} (<1, finite)")


if __name__ == "__main__":
    probe_direction_invariance()
    probe_magnitude_saturation()
    probe_noise_floor()
    probe_length_ceiling()
    probe_count_mechanisms()
    probe_count_v2()
    probe_final()
    probe_ood_length()
    probe_distractor_calibration()
    probe_stability()
    print("\nDone. If P1-P4 read as expected, the core mechanism is sound enough to integrate.")
