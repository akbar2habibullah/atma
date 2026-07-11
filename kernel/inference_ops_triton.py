"""Small forward-only fusions used by the paged inference model."""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _squared_relu_gate_kernel(
        X, GATE, OUT,
        stride_x0, stride_x1, stride_g0, stride_g1,
        N: tl.constexpr, D: tl.constexpr, IS_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        row = idx // D
        col = idx - row * D
        x = tl.load(X + row * stride_x0 + col * stride_x1, mask=mask).to(tl.float32)
        gate = tl.load(GATE + row * stride_g0 + col * stride_g1, mask=mask).to(tl.float32)
        pos = tl.maximum(x, 0.0)
        square = pos * pos
        # Eager BF16 rounds square before the gate multiply.
        if IS_BF16:
            square = square.to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + idx, gate * square, mask=mask)


@torch.no_grad()
def squared_relu_gate(x, gate):
    """Fused gate * relu(x)^2, preserving eager BF16's intermediate rounding."""
    rows, width = x.shape
    out = torch.empty((rows, width), device=x.device, dtype=x.dtype)
    n = rows * width
    _squared_relu_gate_kernel[(triton.cdiv(n, 256),)](
        x, gate, out,
        x.stride(0), x.stride(1), gate.stride(0), gate.stride(1),
        N=n, D=width, IS_BF16=x.dtype == torch.bfloat16, BLOCK=256,
        num_warps=4,
    )
    return out


if HAS_TRITON:

    @triton.jit
    def _softcap_kernel(X, OUT, N: tl.constexpr, IS_BF16: tl.constexpr,
                        BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        x = tl.load(X + idx, mask=mask).to(tl.float32)
        left = 15.0 * x
        square = x * x
        if IS_BF16:
            left = left.to(tl.bfloat16).to(tl.float32)
            square = square.to(tl.bfloat16).to(tl.float32)
        denom = square + 225.0
        if IS_BF16:
            denom = denom.to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(denom)
        if IS_BF16:
            inv = inv.to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + idx, left * inv, mask=mask)


@torch.no_grad()
def softcap_logits(logits):
    """Fuse the elementwise 15*x/sqrt(x^2+225) output soft cap."""
    out = torch.empty_like(logits)
    n = logits.numel()
    _softcap_kernel[(triton.cdiv(n, 256),)](
        logits, out, N=n, IS_BF16=logits.dtype == torch.bfloat16,
        BLOCK=256, num_warps=4,
    )
    return out
