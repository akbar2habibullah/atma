"""Focused CUDA correctness checks not available in the CPU development workspace."""
from __future__ import annotations

import sys
from pathlib import Path


def _tda_reference(q1, q2, k1, k2, v, beta, lambda_param, relu_power):
    import torch
    import torch.nn.functional as F

    q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
    k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)
    length, dim = q1.shape[-2:]
    pos = torch.arange(1, length + 1, device=q1.device, dtype=q1.dtype)
    tau = beta.to(q1.dtype) * torch.sqrt(2 * torch.log(pos) / dim)
    mask = torch.arange(length, device=q1.device)[None, :] <= torch.arange(length, device=q1.device)[:, None]

    def view(q, k):
        score = torch.matmul(q, k.transpose(-2, -1))
        weights = torch.relu(score - tau.view(1, 1, length, 1)).pow(relu_power)
        weights = weights.masked_fill(~mask.view(1, 1, length, length), 0)
        return torch.matmul(weights, v)

    return view(q1, k1) - lambda_param.clamp(0, 1) * view(q2, k2)


def check_tda(source_dir: Path):
    import torch
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from triton_threshold_attention import differential_threshold_rela_triton

    torch.manual_seed(17)
    # B=2 catches flattened batch/head stride errors; T=65 crosses a kernel tile.
    tensors = [torch.randn(2, 2, 65, 64, device="cuda", dtype=torch.float32, requires_grad=True) for _ in range(5)]
    q1, q2, k1, k2, v = tensors
    beta = torch.tensor(1.0, device="cuda")
    lam = torch.tensor(0.5, device="cuda", requires_grad=True)
    out = differential_threshold_rela_triton(q1, q2, k1, k2, v, beta, lam, relu_power=2.0, normalize=True)
    grad = torch.randn_like(out)
    grads = torch.autograd.grad(out, tensors + [lam], grad, retain_graph=True)

    refs = [x.detach().clone().requires_grad_(True) for x in tensors]
    ref_lam = lam.detach().clone().requires_grad_(True)
    ref = _tda_reference(*refs, beta, ref_lam, 2.0)
    ref_grads = torch.autograd.grad(ref, refs + [ref_lam], grad)
    out_err = (out - ref).abs().max().item()
    grad_err = max((a - b).abs().max().item() for a, b in zip(grads, ref_grads))
    if out_err > 2e-4 or grad_err > 2e-3:
        raise RuntimeError(f"TDA kernel parity failed: output max={out_err:.3e}, grad max={grad_err:.3e}")
    print(f"TDA materialized parity OK: output max={out_err:.3e}, grad max={grad_err:.3e}")


__all__ = ["check_tda"]
