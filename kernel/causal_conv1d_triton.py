"""Fused recurrent causal-convolution update for graph-captured decode."""

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
    def _causal_conv1d_step_kernel(
        X, WEIGHT, SLOTS, STATE, OUT,
        stride_xb, stride_xd,
        stride_wd, stride_wk,
        stride_ss, stride_sd, stride_sk,
        stride_ob, stride_od,
        D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        slot = tl.load(SLOTS + b).to(tl.int64)
        x = tl.load(X + b * stride_xb + offs_d * stride_xd, mask=mask)
        base = STATE + slot * stride_ss + offs_d * stride_sd
        out = x * tl.load(WEIGHT + offs_d * stride_wd + (K - 1) * stride_wk, mask=mask)

        # K is 3 or 4 in the model. Loads happen before stores in each unrolled
        # iteration, and the shifted source is never overwritten before use.
        for j in tl.static_range(0, K - 1):
            old = tl.load(base + j * stride_sk, mask=mask)
            out += old * tl.load(WEIGHT + offs_d * stride_wd + j * stride_wk, mask=mask)
            if j > 0:
                tl.store(base + (j - 1) * stride_sk, old, mask=mask)
        tl.store(base + (K - 2) * stride_sk, x, mask=mask)
        tl.store(OUT + b * stride_ob + offs_d * stride_od, out, mask=mask)


@torch.no_grad()
def causal_conv1d_decode_step(x, weight, slots, state_table):
    """Apply one depthwise convolution step and update slot-indexed state in place."""
    B, D = x.shape
    K = weight.shape[1]
    if K < 2:
        return x * weight[:, 0]
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    _causal_conv1d_step_kernel[(B, triton.cdiv(D, 256))](
        x, weight, slots, state_table, out,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        state_table.stride(0), state_table.stride(1), state_table.stride(2),
        out.stride(0), out.stride(1),
        D=D, K=K, BLOCK_D=256, num_warps=4,
    )
    return out
