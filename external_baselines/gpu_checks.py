"""Focused CUDA correctness checks not available in the CPU development workspace."""
from __future__ import annotations

import sys
from pathlib import Path


def _tda_reference(q1, q2, k1, k2, v, beta, lambda_param, relu_power):
    import torch
    import torch.nn.functional as F

    q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
    k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)
    original_dtype = q1.dtype
    # The pinned TDA autograd function performs normalization in the input
    # dtype, then evaluates each threshold-attention path in fp32 and casts its
    # result back before the differential combination.
    if original_dtype != torch.float32:
        q1, q2, k1, k2, v = (x.float() for x in (q1, q2, k1, k2, v))
    length, dim = q1.shape[-2:]
    pos = torch.arange(1, length + 1, device=q1.device, dtype=q1.dtype)
    tau = beta.to(q1.dtype) * torch.sqrt(2 * torch.log(pos) / dim)
    mask = torch.arange(length, device=q1.device)[None, :] <= torch.arange(length, device=q1.device)[:, None]

    def view(q, k):
        score = torch.matmul(q, k.transpose(-2, -1))
        weights = torch.relu(score - tau.view(1, 1, length, 1)).pow(relu_power)
        weights = weights.masked_fill(~mask.view(1, 1, length, length), 0)
        out = torch.matmul(weights, v)
        return out.to(original_dtype) if original_dtype != torch.float32 else out

    return view(q1, k1) - lambda_param.clamp(0, 1) * view(q2, k2)


def check_tda(source_dir: Path):
    import torch
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from triton_threshold_attention import differential_threshold_rela_triton

    torch.manual_seed(17)
    # B=2 catches flattened batch/head stride errors. T=63 and T=65 exercise
    # both sides of the upstream kernel's 64-token tile boundary.
    for dtype in (torch.float32, torch.bfloat16):
        for batch in (1, 2):
            for length in (63, 65):
                tensors = [
                    torch.randn(
                        batch, 2, length, 64, device="cuda", dtype=dtype, requires_grad=True
                    )
                    for _ in range(5)
                ]
                q1, q2, k1, k2, v = tensors
                beta = torch.tensor(1.0, device="cuda")
                lam = torch.tensor(0.5, device="cuda", requires_grad=True)
                out = differential_threshold_rela_triton(
                    q1, q2, k1, k2, v, beta, lam, relu_power=2.0, normalize=True
                )
                grad = torch.randn_like(out)
                grads = torch.autograd.grad(out, tensors + [lam], grad, retain_graph=True)

                refs = [x.detach().clone().requires_grad_(True) for x in tensors]
                ref_lam = lam.detach().clone().requires_grad_(True)
                ref = _tda_reference(*refs, beta, ref_lam, 2.0)
                ref_grads = torch.autograd.grad(ref, refs + [ref_lam], grad)
                out_err = (out - ref).abs().max().item()
                grad_err = max((a - b).abs().max().item() for a, b in zip(grads, ref_grads))
                # The pinned Triton kernel accumulates by 64-token tiles, so
                # fp32 is not bitwise-equivalent to the materialized matmul.
                # These bounds retain a margin over the observed cross-tile
                # error while remaining tight enough to catch stride/mask bugs.
                out_tol, grad_tol = ((1e-3, 1e-2) if dtype == torch.float32 else (2e-3, 2e-2))
                finite = torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
                if not finite or out_err > out_tol or grad_err > grad_tol:
                    raise RuntimeError(
                        "TDA kernel parity failed "
                        f"(dtype={dtype}, batch={batch}, length={length}): "
                        f"finite={bool(finite)}, output max={out_err:.3e}, grad max={grad_err:.3e}"
                    )
                print(
                    "TDA materialized parity OK "
                    f"(dtype={dtype}, batch={batch}, length={length}): "
                    f"output max={out_err:.3e}, grad max={grad_err:.3e}"
                )


__all__ = ["check_tda"]
