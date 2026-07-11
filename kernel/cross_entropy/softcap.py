"""Soft-capped linear cross entropy.

This module follows the Cut Cross-Entropy idea: compute the linear classifier
loss from hidden states and classifier weights without storing the full
``tokens x vocab`` logits matrix. The softcap used by ATMA is folded into the
streaming log-sum-exp:

    z = c * a / sqrt(a^2 + c^2), where a = x @ W.T + b

The current Triton path fuses the forward loss. Backward recomputes raw logits
in bounded chunks, which keeps peak memory tied to ``token_chunk * vocab_chunk``
instead of ``tokens * vocab`` while preserving exact gradients.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover - Triton is optional.
    triton = None
    tl = None
    HAS_TRITON = False


Reduction = Literal["sum", "mean", "none"]
Impl = Literal["auto", "triton", "chunked", "eager"]


def _softcap(raw: Tensor, cap: float) -> Tensor:
    return cap * raw * torch.rsqrt(raw.square() + cap * cap)


def _softcap_grad(raw: Tensor, cap: float) -> Tensor:
    denom = raw.square() + cap * cap
    return (cap * cap * cap) * torch.rsqrt(denom * denom * denom)


def _flatten_inputs(x: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError(f"x must have shape (..., hidden), got {tuple(x.shape)}")
    if targets.shape != x.shape[:-1]:
        raise ValueError(
            f"targets must have shape {tuple(x.shape[:-1])}, got {tuple(targets.shape)}"
        )
    return x.reshape(-1, x.shape[-1]), targets.reshape(-1).long(), targets.shape


def _validate_targets(targets: Tensor, vocab_size: int, ignore_index: int) -> None:
    active = targets != ignore_index
    if active.any():
        bad = active & ((targets < 0) | (targets >= vocab_size))
        if bad.any():
            first = targets[bad][0].item()
            raise ValueError(f"target id {first} is outside vocab size {vocab_size}")


def softcap_cross_entropy_reference(
    x: Tensor,
    weight: Tensor,
    targets: Tensor,
    bias: Tensor | None = None,
    *,
    softcap: float = 15.0,
    reduction: Reduction = "sum",
    ignore_index: int = -100,
) -> Tensor:
    """Reference implementation that materializes logits.

    Use this for parity tests and CPU baselines, not for memory-sensitive
    training.
    """

    x2d, targets1d, target_shape = _flatten_inputs(x, targets)
    logits = F.linear(x2d, weight.type_as(x2d), None if bias is None else bias.type_as(x2d))
    logits = _softcap(logits.float(), softcap)
    loss = F.cross_entropy(logits, targets1d, reduction=reduction, ignore_index=ignore_index)
    if reduction == "none":
        return loss.reshape(target_shape)
    return loss


def _chunked_forward(
    x2d: Tensor,
    weight: Tensor,
    targets: Tensor,
    bias: Tensor | None,
    *,
    softcap: float,
    ignore_index: int,
    vocab_chunk_size: int,
) -> tuple[Tensor, Tensor]:
    n_tokens, hidden = x2d.shape
    vocab_size = weight.shape[0]
    device = x2d.device

    losses = torch.zeros(n_tokens, device=device, dtype=torch.float32)
    lse = torch.zeros(n_tokens, device=device, dtype=torch.float32)
    valid = targets != ignore_index
    if n_tokens == 0:
        return losses, lse

    x_f = x2d.float()
    weight_f = weight.float()
    bias_f = None if bias is None else bias.float()

    running_m = torch.full((n_tokens,), -torch.inf, device=device, dtype=torch.float32)
    running_l = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
    target_z = torch.zeros((n_tokens,), device=device, dtype=torch.float32)

    for v0 in range(0, vocab_size, vocab_chunk_size):
        v1 = min(v0 + vocab_chunk_size, vocab_size)
        raw = x_f @ weight_f[v0:v1].T
        if bias_f is not None:
            raw = raw + bias_f[v0:v1]
        z = _softcap(raw, softcap)

        block_m = z.max(dim=1).values
        m_new = torch.maximum(running_m, block_m)
        running_l = running_l * torch.exp(running_m - m_new) + torch.exp(z - m_new[:, None]).sum(dim=1)
        running_m = m_new

        in_block = valid & (targets >= v0) & (targets < v1)
        if in_block.any():
            rows = in_block.nonzero(as_tuple=False).flatten()
            cols = targets[rows] - v0
            target_z[rows] = z[rows, cols]

    lse = running_m + running_l.log()
    losses[valid] = lse[valid] - target_z[valid]
    return losses, lse


if HAS_TRITON:

    @triton.jit
    def _softcap_xent_fwd_kernel(
        X,
        W,
        BIAS,
        TARGETS,
        LOSSES,
        LSE,
        N_TOKENS: tl.constexpr,
        HIDDEN: tl.constexpr,
        VOCAB: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_v = tl.arange(0, BLOCK_V)
        offs_d = tl.arange(0, BLOCK_D)
        target = tl.load(TARGETS + row)
        active = target != IGNORE_INDEX

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        z_y = tl.full((), 0.0, tl.float32)

        for v0 in range(0, VOCAB, BLOCK_V):
            v = v0 + offs_v
            logits = tl.zeros((BLOCK_V,), tl.float32)

            for d0 in range(0, HIDDEN, BLOCK_D):
                d = d0 + offs_d
                x = tl.load(X + row * HIDDEN + d, mask=d < HIDDEN, other=0.0).to(tl.float32)
                w = tl.load(
                    W + v[:, None] * HIDDEN + d[None, :],
                    mask=(v[:, None] < VOCAB) & (d[None, :] < HIDDEN),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(w * x[None, :], axis=1)

            if HAS_BIAS:
                b = tl.load(BIAS + v, mask=v < VOCAB, other=0.0).to(tl.float32)
                logits += b

            cap = tl.full((), SOFTCAP, tl.float32)
            z = cap * logits * tl.rsqrt(logits * logits + cap * cap)
            z = tl.where(v < VOCAB, z, -float("inf"))

            block_m = tl.max(z, axis=0)
            m_new = tl.maximum(m_i, block_m)
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(z - m_new), axis=0)
            m_i = m_new

            hit = active & (target >= v0) & (target < v0 + BLOCK_V)
            z_hit = tl.sum(tl.where(v == target, z, 0.0), axis=0)
            z_y = tl.where(hit, z_hit, z_y)

        lse = m_i + tl.log(l_i)
        loss = lse - z_y
        loss = tl.where(active, loss, 0.0)
        lse = tl.where(active, lse, 0.0)
        tl.store(LOSSES + row, loss)
        tl.store(LSE + row, lse)


def _triton_forward(
    x2d: Tensor,
    weight: Tensor,
    targets: Tensor,
    bias: Tensor | None,
    *,
    softcap: float,
    ignore_index: int,
    vocab_chunk_size: int,
    hidden_block_size: int,
) -> tuple[Tensor, Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton softcap cross entropy requested but Triton/CUDA is unavailable")
    if not x2d.is_cuda:
        raise RuntimeError("Triton softcap cross entropy requires CUDA tensors")

    n_tokens, hidden = x2d.shape
    vocab_size = weight.shape[0]
    losses = torch.empty(n_tokens, device=x2d.device, dtype=torch.float32)
    lse = torch.empty(n_tokens, device=x2d.device, dtype=torch.float32)
    if n_tokens == 0:
        return losses, lse

    block_v = triton.next_power_of_2(vocab_chunk_size)
    block_d = triton.next_power_of_2(hidden_block_size)
    _softcap_xent_fwd_kernel[(n_tokens,)](
        x2d,
        weight,
        x2d if bias is None else bias,
        targets,
        losses,
        lse,
        n_tokens,
        hidden,
        vocab_size,
        float(softcap),
        int(ignore_index),
        bias is not None,
        block_v,
        block_d,
        num_warps=8,
        num_stages=3,
    )
    return losses, lse


def _chunked_backward(
    grad_out: Tensor,
    x2d: Tensor,
    weight: Tensor,
    targets: Tensor,
    bias: Tensor | None,
    lse: Tensor,
    *,
    softcap: float,
    reduction: Reduction,
    ignore_index: int,
    target_shape: torch.Size,
    token_chunk_size: int,
    vocab_chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    n_tokens, hidden = x2d.shape
    vocab_size = weight.shape[0]
    valid = targets != ignore_index

    if reduction == "none":
        row_scale = grad_out.reshape(-1).float()
    else:
        row_scale = grad_out.float().expand(n_tokens).clone()
        if reduction == "mean":
            denom = valid.count_nonzero().clamp_min(1).to(torch.float32)
            row_scale = row_scale / denom
    row_scale = torch.where(valid, row_scale, torch.zeros_like(row_scale))

    grad_x = torch.zeros((n_tokens, hidden), device=x2d.device, dtype=torch.float32)
    grad_w = torch.zeros_like(weight, dtype=torch.float32)
    grad_b = None if bias is None else torch.zeros_like(bias, dtype=torch.float32)

    x_f = x2d.float()
    weight_f = weight.float()
    bias_f = None if bias is None else bias.float()

    for t0 in range(0, n_tokens, token_chunk_size):
        t1 = min(t0 + token_chunk_size, n_tokens)
        x_blk = x_f[t0:t1]
        y_blk = targets[t0:t1]
        scale_blk = row_scale[t0:t1]
        lse_blk = lse[t0:t1]
        gx_blk = torch.zeros_like(x_blk)

        if not torch.any(scale_blk != 0):
            continue

        for v0 in range(0, vocab_size, vocab_chunk_size):
            v1 = min(v0 + vocab_chunk_size, vocab_size)
            w_blk = weight_f[v0:v1]
            raw = x_blk @ w_blk.T
            if bias_f is not None:
                raw = raw + bias_f[v0:v1]

            z = _softcap(raw, softcap)
            grad_z = torch.exp(z - lse_blk[:, None])
            in_block = (y_blk >= v0) & (y_blk < v1)
            if in_block.any():
                rows = in_block.nonzero(as_tuple=False).flatten()
                cols = y_blk[rows] - v0
                grad_z[rows, cols] -= 1.0

            grad_raw = grad_z * _softcap_grad(raw, softcap)
            grad_raw = grad_raw * scale_blk[:, None]

            gx_blk += grad_raw @ w_blk
            grad_w[v0:v1] += grad_raw.T @ x_blk
            if grad_b is not None:
                grad_b[v0:v1] += grad_raw.sum(dim=0)

        grad_x[t0:t1] = gx_blk

    return grad_x.reshape(*target_shape, hidden).to(x2d.dtype), grad_w.to(weight.dtype), (
        None if grad_b is None else grad_b.to(bias.dtype)
    )


class _SoftcapLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        targets: Tensor,
        softcap: float,
        reduction: str,
        ignore_index: int,
        impl: str,
        token_chunk_size: int,
        vocab_chunk_size: int,
        hidden_block_size: int,
    ) -> Tensor:
        x2d, targets1d, target_shape = _flatten_inputs(x, targets)
        _validate_targets(targets1d, weight.shape[0], ignore_index)
        if weight.ndim != 2:
            raise ValueError(f"weight must have shape (vocab, hidden), got {tuple(weight.shape)}")
        if weight.shape[1] != x2d.shape[1]:
            raise ValueError(f"weight hidden dim {weight.shape[1]} != x hidden dim {x2d.shape[1]}")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError(f"bias must have shape ({weight.shape[0]},), got {tuple(bias.shape)}")

        x2d_c = x2d.contiguous()
        weight_c = weight.contiguous()
        bias_c = None if bias is None else bias.contiguous()
        targets_c = targets1d.contiguous()

        use_triton = impl == "triton" or (impl == "auto" and x2d_c.is_cuda and HAS_TRITON)
        if use_triton:
            losses, lse = _triton_forward(
                x2d_c,
                weight_c,
                targets_c,
                bias_c,
                softcap=softcap,
                ignore_index=ignore_index,
                vocab_chunk_size=vocab_chunk_size,
                hidden_block_size=hidden_block_size,
            )
        else:
            losses, lse = _chunked_forward(
                x2d_c,
                weight_c,
                targets_c,
                bias_c,
                softcap=softcap,
                ignore_index=ignore_index,
                vocab_chunk_size=vocab_chunk_size,
            )

        ctx.save_for_backward(x2d_c, weight_c, targets_c, lse, torch.empty(0, device=x.device) if bias_c is None else bias_c)
        ctx.has_bias = bias_c is not None
        ctx.softcap = float(softcap)
        ctx.reduction = reduction
        ctx.ignore_index = int(ignore_index)
        ctx.target_shape = target_shape
        ctx.token_chunk_size = int(token_chunk_size)
        ctx.vocab_chunk_size = int(vocab_chunk_size)

        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            denom = (targets_c != ignore_index).count_nonzero()
            return losses.sum() / denom
        return losses.reshape(target_shape)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x2d, weight, targets, lse, bias_saved = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        grad_x, grad_w, grad_b = _chunked_backward(
            grad_out,
            x2d,
            weight,
            targets,
            bias,
            lse,
            softcap=ctx.softcap,
            reduction=ctx.reduction,
            ignore_index=ctx.ignore_index,
            target_shape=ctx.target_shape,
            token_chunk_size=ctx.token_chunk_size,
            vocab_chunk_size=ctx.vocab_chunk_size,
        )
        return grad_x, grad_w, grad_b, None, None, None, None, None, None, None, None


def softcap_linear_cross_entropy(
    x: Tensor,
    weight: Tensor,
    targets: Tensor,
    bias: Tensor | None = None,
    *,
    softcap: float = 15.0,
    reduction: Reduction = "sum",
    ignore_index: int = -100,
    impl: Impl = "auto",
    token_chunk_size: int = 128,
    vocab_chunk_size: int = 2048,
    hidden_block_size: int = 64,
) -> Tensor:
    """Compute soft-capped linear cross entropy without storing full logits.

    Args:
        x: Hidden states shaped ``(..., hidden)``.
        weight: Classifier matrix shaped ``(vocab, hidden)``.
        targets: Class ids shaped like ``x.shape[:-1]``.
        bias: Optional classifier bias shaped ``(vocab,)``.
        softcap: Saturating logit cap. ATMA uses ``15.0``.
        reduction: ``"sum"``, ``"mean"``, or ``"none"``.
        impl: ``"auto"`` uses Triton forward on CUDA and chunked torch otherwise.
            ``"chunked"`` always uses the portable chunked implementation.
            ``"eager"`` materializes logits and is intended only as a reference.
        token_chunk_size: Number of tokens per backward recompute block.
        vocab_chunk_size: Vocabulary columns per streamed/recomputed block.
        hidden_block_size: Hidden columns per Triton forward dot block.
    """

    if reduction not in ("sum", "mean", "none"):
        raise ValueError(f"unsupported reduction {reduction!r}")
    if impl not in ("auto", "triton", "chunked", "eager"):
        raise ValueError(f"unsupported impl {impl!r}")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if token_chunk_size <= 0 or vocab_chunk_size <= 0 or hidden_block_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if impl == "eager":
        return softcap_cross_entropy_reference(
            x,
            weight,
            targets,
            bias,
            softcap=softcap,
            reduction=reduction,
            ignore_index=ignore_index,
        )
    if impl == "triton" and (not HAS_TRITON or not x.is_cuda):
        raise RuntimeError("impl='triton' requires CUDA tensors and Triton")

    return _SoftcapLinearCrossEntropy.apply(
        x,
        weight,
        bias,
        targets,
        float(softcap),
        reduction,
        int(ignore_index),
        impl,
        int(token_chunk_size),
        int(vocab_chunk_size),
        int(hidden_block_size),
    )
