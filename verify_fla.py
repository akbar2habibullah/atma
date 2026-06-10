"""GPU check: FLA chunk_gated_delta_rule path vs the validated torch reference (TitansMemory).

Run on the pod:  python verify_fla.py

The torch reference now uses FLA's exact convention (decay-first, undecayed write,
self-inclusive readout M_t q_t), so the recurrences MATCH. The only residual is numerics:
FLA runs the kernel in bf16, the torch reference computes in fp32, and L2-norm is applied
in-kernel vs via F.normalize. So:

  rel_err  < ~0.05 -> correct (bf16-vs-fp32 numerics only). Ship the FLA path.
  rel_err  > ~0.10 -> a real residual mismatch (g space / beta range / layout / scale).
                      Send the numbers and I'll fix it.

Also smoke-checks: outputs finite, gate-head gradients finite & non-trivial.
"""

import sys
import torch
import torch.nn.functional as F

import model.blocks as mb
from model.blocks import TitansMemory


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def verify_inference_bridge():
    """Checks the exact FLA usage in inference/models/atma.py (the paged-engine memory
    branch): chunked-prefill state carry (initial_state/output_final_state), the state
    LAYOUT bridge (FLA [K,V] vs torch gated_delta_chunked's transposed [V,K]), and
    chunk-prefill -> fused_recurrent decode continuity. All single-source-of-truth
    against one full chunked pass and the fp32 torch oracle.

      rel_err < ~0.05 -> correct (bf16-vs-fp32 / kernel-boundary numerics only).
      rel_err > ~0.10 -> real mismatch (layout / state convention) — do NOT ship decode.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from model.blocks import gated_delta_chunked

    print("\n── inference bridge (paged-engine mem branch) ──")
    torch.manual_seed(1)
    dev = "cuda"
    B, H, T, dk, Tdec = 2, 8, 256, 128, 8
    Ttot = T + Tdec
    dtype = torch.bfloat16

    q = torch.randn(B, Ttot, H, dk, device=dev, dtype=dtype)
    k = torch.randn(B, Ttot, H, dk, device=dev, dtype=dtype)
    v = torch.randn(B, Ttot, H, dk, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, Ttot, H, device=dev) + 3.9)      # log-decay, fp32
    beta = torch.sigmoid(torch.randn(B, Ttot, H, device=dev))        # fp32
    kw = dict(scale=1.0, use_qk_l2norm_in_kernel=True)

    # NOTE: g and beta are keyword-bound everywhere. fused_recurrent_gated_delta_rule has
    # extra per-key/value gate params (gk, gv) between g and beta, so a positional beta
    # binds to gk and is indexed as [B, T, H, K] -> out-of-bounds reads (NaN at small B,
    # illegal memory access at large B). chunk_gated_delta_rule has no gk/gv today, but
    # keywords are version-proof.
    def fla_chunk(s, e, S0=None):
        return chunk_gated_delta_rule(q=q[:, s:e].contiguous(), k=k[:, s:e].contiguous(),
                                      v=v[:, s:e].contiguous(), g=g[:, s:e].contiguous(),
                                      beta=beta[:, s:e].contiguous(),
                                      initial_state=S0, output_final_state=True, **kw)

    # oracle: one chunked pass over the whole stream
    o_full, _ = fla_chunk(0, Ttot)

    # 1) chunked-prefill state carry (split at a non-multiple of the kernel chunk)
    T1 = 100
    _, S_a = fla_chunk(0, T1)
    o_b, _ = fla_chunk(T1, T, S0=S_a)
    print(f"chunk-split state carry   rel_err = {rel(o_b.float(), o_full[:, T1:T].float()):.4f}")

    # 2) layout bridge: FLA final state == transpose of the torch oracle's state
    o_pre, S_pre = fla_chunk(0, T)
    qn = F.normalize(q[:, :T].float().transpose(1, 2), dim=-1)       # (B, H, T, dk) unit
    kn = F.normalize(k[:, :T].float().transpose(1, 2), dim=-1)
    r_t, S_t = gated_delta_chunked(qn, kn, v[:, :T].float().transpose(1, 2),
                                   torch.exp(g[:, :T]).transpose(1, 2),
                                   beta[:, :T].transpose(1, 2), chunk=64)
    print(f"prefill out vs torch      rel_err = {rel(o_pre.float(), r_t.transpose(1, 2)):.4f}")
    print(f"state layout vs torch     rel_err = {rel(S_pre.float(), S_t.transpose(-1, -2)):.4f}"
          f"   (transposed-compare control: {rel(S_pre.float(), S_t):.4f} — should be MUCH larger)")

    # 3) decode continuity: fused_recurrent single-token steps from the prefill state
    S = S_pre
    outs = []
    for t in range(T, Ttot):
        o_t, S = fused_recurrent_gated_delta_rule(
            q=q[:, t:t + 1].contiguous(), k=k[:, t:t + 1].contiguous(),
            v=v[:, t:t + 1].contiguous(), g=g[:, t:t + 1].contiguous(),
            beta=beta[:, t:t + 1].contiguous(),
            initial_state=S, output_final_state=True, **kw)
        outs.append(o_t)
    o_dec = torch.cat(outs, dim=1)
    print(f"chunk->recurrent decode   rel_err = {rel(o_dec.float(), o_full[:, T:].float()):.4f}")

    # 4) torch fallback decode step (the CPU path in _mem_decode) vs FLA recurrent step
    gamma1 = torch.exp(g[:, T])                                       # (B, H)
    beta1 = beta[:, T]
    q1 = F.normalize(q[:, T].float(), dim=-1)                         # (B, H, dk)
    k1 = F.normalize(k[:, T].float(), dim=-1)
    St = S_pre.float().transpose(-1, -2)                              # torch (B, H, dv, dk)
    Sd = gamma1[..., None, None] * St
    u = beta1[..., None] * (v[:, T].float() - torch.einsum("bhvk,bhk->bhv", Sd, k1))
    S_new = Sd + u.unsqueeze(-1) * k1.unsqueeze(-2)
    r1 = torch.einsum("bhvk,bhk->bhv", S_new, q1)
    print(f"torch decode step vs FLA  rel_err = {rel(outs[0].squeeze(1).float(), r1):.4f}")


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA"); return
    if not mb._HAS_FLA:
        print("SKIP: flash-linear-attention not importable (pip install flash-linear-attention)"); return

    torch.manual_seed(0)
    dev = "cuda"
    B, H, T, dk, D = 2, 8, 512, 128, 1024
    dtype = torch.bfloat16

    from train.model import Linear   # dtype-casting Linear used in training
    mem = TitansMemory(D, H, dk, Linear, chunk=128).to(dev)
    # activate the zero-init readout so the branch (and its grads) are non-trivial
    torch.nn.init.normal_(mem.proj.weight, std=0.02)

    x = torch.randn(B, T, D, device=dev, dtype=dtype, requires_grad=True)
    q = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, dk, device=dev, dtype=dtype, requires_grad=True)

    def run(kernel):
        mem.kernel = kernel
        xi, qi, ki, vi = (t.detach().clone().requires_grad_(True) for t in (x, q, k, v))
        out = mem(xi, qi, ki, vi)
        out.float().pow(2).mean().backward()
        return out.detach(), {"x": xi.grad, "q": qi.grad, "k": ki.grad,
                              "w_gamma": mem.w_gamma.weight.grad.clone(),
                              "w_beta": mem.w_beta.weight.grad.clone()}

    o_fla, g_fla = run("fla")
    mem.zero_grad()
    o_torch, g_torch = run("torch")

    print(f"shapes B={B} H={H} T={T} dk={dk} dtype={dtype}")
    print(f"output  rel_err = {rel(o_fla.float(), o_torch.float()):.4f}")
    for name in g_fla:
        print(f"  grad[{name:>8}] rel_err = {rel(g_fla[name].float(), g_torch[name].float()):.4f}")
    finite = torch.isfinite(o_fla).all().item() and all(torch.isfinite(g).all().item() for g in g_fla.values())
    nontrivial = g_fla["w_gamma"].norm().item() > 0
    print(f"FLA finite={finite}  w_gamma grad nonzero={nontrivial}")
    print("\nInterpretation: output rel_err < ~0.05 => mapping correct (residual = readout"
          " self-term). > ~0.15 => API mismatch, send me these numbers.")

    verify_inference_bridge()


if __name__ == "__main__":
    main()
