"""GPU check: FLA chunk_gated_delta_rule path vs the validated torch reference (TitansMemory).

Run on the pod:  python verify_fla.py

The FLA path is mapped to our recurrence by value pre-scaling (v <- gamma*v, so FLA's
beta*v write becomes our gamma*beta*v) + g=log(gamma) + in-kernel L2-norm. The one
expected residual is the readout self-term convention (FLA includes the current token in
o_t; the torch reference uses M_{t-1} q_t, strictly causal). So:

  rel_err  < ~5%   -> mapping is right; residual is the self-term convention (fine to ship).
  rel_err  > ~15%  -> likely an API mismatch (g space / beta range / layout). Send the
                      numbers and I'll fix the mapping.

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
