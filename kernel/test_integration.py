"""End-to-end integration test: train.model PolarAttention / Model with the
Triton kernel selected (attn_kernel='triton') vs the torch path, on CUDA.

    python -m kernel.test_integration
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

from model.config import AtmaConfig
import train.model as tm

# force pure-pytorch causal conv (no hub download), keeps paths comparable
tm.causal_conv1d_fn = tm._causal_conv1d_fallback

torch.manual_seed(0)
dev = "cuda"
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {extra}" if extra else ""))
    PASS += bool(cond); FAIL += (not cond)


def copy(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing and not unexpected, f"{missing} {unexpected}"


dim, hd, nkv, ks = 512, 128, 1, 4

# ── layer parity: triton vs online (fp32) ─────────────────────────────────
print("\n── PolarAttention(attn_kernel='triton') vs online (fp32) ──")
torch.set_default_dtype(torch.float32)
a_tri = tm.PolarAttention(dim, head_dim=hd, num_kv_heads=nkv, num_random_keys=0, kernel_size=ks,
                          attn_kernel="triton").to(dev)
a_on = tm.PolarAttention(dim, head_dim=hd, num_kv_heads=nkv, num_random_keys=0, kernel_size=ks,
                         online=True, k_block=128, attn_kernel="torch").to(dev)
copy(a_tri, a_on)

x1 = torch.randn(2, 192, dim, device=dev, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)
y1, _ = a_tri(x1)
y2, _ = a_on(x2)
y1.float().pow(2).mean().backward()
y2.float().pow(2).mean().backward()

dy = (y1 - y2).abs().max().item()
dx = (x1.grad - x2.grad).abs().max().item()
check("forward output", dy < 1e-4, f"max_diff={dy:.2e}")
check("grad to input x", dx < 1e-4, f"max_diff={dx:.2e}")
for n, p_tri in a_tri.named_parameters():
    p_on = dict(a_on.named_parameters())[n]
    if p_tri.grad is None:
        continue
    d = (p_tri.grad - p_on.grad).abs().max().item()
    rel = d / (p_on.grad.abs().max().item() + 1e-9)
    check(f"grad {n}", rel < 5e-3, f"abs={d:.2e} rel={rel:.2e}")

# ── full Model forward+backward finite (triton, bf16) ─────────────────────
print("\n── full train Model (attn_kernel='triton') fwd+bwd finite ──")
cfg = AtmaConfig(vocab_size=256, num_hidden_layers=4, hidden_size=dim, head_dim=hd,
                 num_random_keys=8, attn_kernel="triton")
m = tm.Model(cfg).to(dev)
m.train()
ii = torch.randint(0, 256, (2, 128), device=dev)
tt = torch.randint(0, 256, (2, 128), device=dev)
task, reg, align = m(ii, tt)
(task + reg + align).backward()
grads = [p.grad for p in m.parameters() if p.grad is not None]
all_finite = all(torch.isfinite(g).all() for g in grads)
lg = [b.attn.len_gain_raw.grad for b in m.blocks if isinstance(b.attn, tm.PolarAttention)]
lg_ok = all(g is not None and torch.isfinite(g).all() for g in lg)
check("Model backward all-finite", all_finite and lg_ok,
      f"task={float(task):.1f} align={float(align):.4f}")

print(f"\n{'='*50}\nResults: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
