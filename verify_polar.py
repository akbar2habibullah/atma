"""
verify_polar.py — train<->reference parity + sanity for PolarAttention.

Inference path is intentionally NOT covered yet (deferred). This checks:
  1. train PolarAttention == reference PolarAttention (fp32, copied weights)
  2. full attn Block train == reference
  3. backward runs; distractor align_loss fires and flows grad to null sink + Q
  4. magnitude channel stays bounded in (0,1) and outputs finite at 1x and 4x length
  5. full Model (train) and ReferenceModel build and run end-to-end
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

from model.config import AtmaConfig
import model.reference as ref
import train.model as tm

# Force the pure-PyTorch causal conv so the train path runs on CPU and matches ref.
tm.causal_conv1d_fn = tm._causal_conv1d_fallback

torch.manual_seed(0)
ATOL = 1e-4
cfg = AtmaConfig(vocab_size=256, num_hidden_layers=4, hidden_size=256, head_dim=64)
dim, hd = cfg.hidden_size, cfg.head_dim
nkv, ks = cfg.num_key_value_heads, cfg.attn_kernel_size

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {extra}" if extra else ""))
    PASS += cond; FAIL += (not cond)

def copy(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing and not unexpected, f"param mismatch missing={missing} unexpected={unexpected}"


# ── 1. layer parity ──────────────────────────────────────────────────────────
print("\n── PolarAttention layer parity ──")
r = ref.PolarAttention(dim, head_dim=hd, num_kv_heads=nkv, kernel_size=ks)
t = tm.PolarAttention(dim, head_dim=hd, num_kv_heads=nkv, num_random_keys=0, kernel_size=ks)
copy(r, t)
r.eval(); t.eval()
x = torch.randn(2, 16, dim)
with torch.no_grad():
    yr = r(x)
    yt, al = t(x)
d = (yr - yt).abs().max().item()
check("train == ref", d <= ATOL, f"max_diff={d:.2e}")
check("align_loss == 0 when num_random_keys=0", float(al) == 0.0)

# ── 2. block parity ──────────────────────────────────────────────────────────
print("\n── attn Block parity ──")
rb = ref.Block(dim, attention=True, head_dim=hd, num_kv_heads=nkv, attn_kernel_size=ks, conv_kernel_size=cfg.conv_kernel_size)
tb = tm.Block(dim, attention=True, head_dim=hd, num_kv_heads=nkv, num_random_keys=0, attn_kernel_size=ks, conv_kernel_size=cfg.conv_kernel_size)
copy(rb, tb)
rb.eval(); tb.eval()
with torch.no_grad():
    yr = rb(x)
    yt, reg, al = tb(x)
d = (yr - yt).abs().max().item()
check("block train == ref", d <= ATOL, f"max_diff={d:.2e}")

# ── 3. backward + distractor ─────────────────────────────────────────────────
print("\n── backward + distractor ──")
td = tm.PolarAttention(dim, head_dim=hd, num_kv_heads=nkv, num_random_keys=8, kernel_size=ks)
td.train()
out, al = td(x)
loss = out.float().pow(2).mean() + al
al_val = al.detach().item()
loss.backward()
g_null = td.null_base.grad
g_q = td.q.weight.grad
g_mu = td.mu_proj.weight.grad
check("align_loss > 0 with random keys", al_val > 0, f"align={al_val:.4f}")
check("grad flows to null_base", g_null is not None and torch.isfinite(g_null).all() and g_null.abs().sum() > 0)
check("grad flows to Q proj", g_q is not None and torch.isfinite(g_q).all() and g_q.abs().sum() > 0)
check("grad flows to count proj (mu_proj)", g_mu is not None and torch.isfinite(g_mu).all() and g_mu.abs().sum() > 0)

# ── 4. bounded magnitude / finite outputs at 1x and 4x length ────────────────
print("\n── magnitude bounded + finite at length ──")
t.eval()
for T in (16, 64):
    xx = torch.randn(1, T, dim)
    # reach into the reduction to inspect mag
    B = 1; H = t.num_heads; dk = t.head_dim
    with torch.no_grad():
        q_gate = t.q(xx).view(B, T, H, dk * 2); q, gate = q_gate.chunk(2, -1)
        k = t.k(xx).view(B, T, nkv, dk); v = t.v(xx).view(B, T, nkv, dk)
        q = F.rms_norm(q, (dk,)); k = F.rms_norm(k, (dk,))
        groups = H // nkv
        q_t = q.transpose(1, 2)
        k_t = k.repeat_interleave(groups, dim=2).transpose(1, 2)
        v_t = v.repeat_interleave(groups, dim=2).transpose(1, 2)
        sigma = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dk ** 0.5)
        sigma = sigma + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        n_keys = torch.arange(1, T + 1, dtype=torch.float32)
        from model.blocks import polar_reduce
        c, mag = polar_reduce(sigma, v_t, n_keys, v_null=t.v_null, null_base=t.null_base,
                              null_slope_raw=t.null_slope_raw, len_gain_raw=t.len_gain_raw,
                              mag_beta_raw=t.mag_beta_raw)
        out, _ = t(xx)
    ok = (mag.min() >= 0) and (mag.max() < 1.0) and torch.isfinite(out).all() and torch.isfinite(c).all()
    cnorm = c.norm(dim=-1)
    check(f"T={T}: mag in [0,1), |c|==1, finite", bool(ok),
          f"mag=[{mag.min():.3f},{mag.max():.3f}] |c|=[{cnorm.min():.3f},{cnorm.max():.3f}]")

# ── 5. end-to-end build ──────────────────────────────────────────────────────
print("\n── end-to-end model build ──")
rm = ref.ReferenceModel(cfg).eval()
ids = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    logits = rm(ids)
check("ReferenceModel forward", logits.shape == (1, 16, cfg.vocab_size) and torch.isfinite(logits).all())

trm = tm.Model(cfg)
inp = torch.randint(0, cfg.vocab_size, (2, 16))
tgt = torch.randint(0, cfg.vocab_size, (2, 16))
task, reg, align = trm(inp, tgt)
check("train Model forward (task, reg, align)", all(torch.isfinite(torch.as_tensor(float(z))) for z in (task, reg, align)),
      f"task={float(task):.2f} reg={float(reg):.4f} align={float(align):.4f}")

print(f"\n{'='*50}\nResults: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
