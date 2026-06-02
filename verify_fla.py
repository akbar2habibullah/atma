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


if __name__ == "__main__":
    main()
