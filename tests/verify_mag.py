"""Verify the MAG memory integration (Step 3 of plans/linked-forging-sparrow.md).

Checks that the train and reference PolarAttention agree once the Titans memory branch
and the trainable sliding window are enabled -- the memory branch is pure PyTorch and
identical in both, so train == reference must still hold (the repo's standing invariant).

  A regression : mem off, window None  -> train == ref (unchanged from plain polar)
  B window      : mem off, window=W     -> train == ref (windowed core in the model)
  C MAG active  : mem on,  window=W, mem.proj randomized -> train == ref AND != B
  D safe no-op  : mem on,  window=W, mem.proj zero-init (default) -> output == B
"""

import sys
import torch

import model.reference as ref_mod
import train.model as tm

tm.causal_conv1d_fn = tm._causal_conv1d_fallback  # pure-PyTorch conv so train runs on CPU

DIM, HD, NKV, K, T, W = 128, 64, 1, 4, 24, 8     # 2 heads, 1 KV head; window 8 < T
ATOL = 1e-5
FAILS = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def _build(mem_enabled, window):
    ref = ref_mod.PolarAttention(DIM, head_dim=HD, num_kv_heads=NKV, kernel_size=K,
                                 window=window, mem_enabled=mem_enabled, mem_chunk=8)
    train = tm.PolarAttention(DIM, head_dim=HD, num_kv_heads=NKV, num_random_keys=0, kernel_size=K,
                              window=window, mem_enabled=mem_enabled, mem_chunk=8)
    return ref, train


def _copy(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing and not unexpected, f"param mismatch: missing={missing} unexpected={unexpected}"


def _fwd(ref, train, x):
    with torch.no_grad():
        y_ref = ref(x)
        y_train, _ = train(x)                 # train PolarAttention returns (out, align_loss)
    return y_ref, y_train


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, T, DIM, dtype=torch.float32)

    # A — regression: no memory, full attention
    ref, train = _build(mem_enabled=False, window=None)
    _copy(ref, train)
    yr, yt = _fwd(ref, train, x)
    _check("A regression (mem off, full)", (yr - yt).abs().max() < ATOL,
           f"max_diff={(yr - yt).abs().max():.2e}")

    # B — windowed core, no memory
    ref, train = _build(mem_enabled=False, window=W)
    _copy(ref, train)
    yr_B, yt_B = _fwd(ref, train, x)
    _check("B window (mem off, W=8)", (yr_B - yt_B).abs().max() < ATOL,
           f"max_diff={(yr_B - yt_B).abs().max():.2e}")

    # C — MAG with an ACTIVE memory branch (randomize the zero-init readout proj).
    # Compare mem-on vs mem-off WITHIN the same instance (toggle .mem) so the base
    # polar weights are identical and the diff is purely the memory contribution.
    ref, train = _build(mem_enabled=True, window=W)
    torch.nn.init.normal_(ref.mem.proj.weight, std=0.05)
    torch.nn.init.normal_(ref.mem.proj.bias, std=0.05)
    _copy(ref, train)                          # train gets the same randomized proj
    yr_C, yt_C = _fwd(ref, train, x)
    _check("C MAG active: train == ref", (yr_C - yt_C).abs().max() < ATOL,
           f"max_diff={(yr_C - yt_C).abs().max():.2e}")
    mem_mod = ref.mem
    ref.mem = None
    with torch.no_grad():
        yr_C_off = ref(x)
    ref.mem = mem_mod
    _check("C MAG active: memory changes output", (yr_C - yr_C_off).abs().max() > 1e-3,
           f"||mem contribution||_inf={(yr_C - yr_C_off).abs().max():.2e}")

    # D — safe no-op: default zero-init proj => memory contributes nothing at init
    # (mem-on with zero proj == mem-branch removed, same base weights).
    ref, train = _build(mem_enabled=True, window=W)
    _copy(ref, train)
    with torch.no_grad():
        yr_D_on = ref(x)                       # mem on, proj zero-init
        mem_mod = ref.mem
        ref.mem = None
        yr_D_off = ref(x)                      # mem branch removed
        ref.mem = mem_mod
    _check("D safe no-op: zero-init mem is a no-op", (yr_D_on - yr_D_off).abs().max() < ATOL,
           f"max_diff={(yr_D_on - yr_D_off).abs().max():.2e}")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        sys.exit(1)
    print("ALL PASS")
