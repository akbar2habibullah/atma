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
    def _packed_causal_conv1d_kernel(
        X, WEIGHT, OUT, STATE, TOKEN_SEQ_START, TOKEN_SEQ_END, TOKEN_SEQ_SLOT,
        stride_xt, stride_xc, stride_wc, stride_wk, stride_ot, stride_oc,
        stride_ss, stride_sc, stride_sk, TOTAL, CHANNELS,
        KERNEL_SIZE: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        token_valid = offs_t < TOTAL
        channel_valid = offs_c < CHANNELS
        seq_start = tl.load(TOKEN_SEQ_START + offs_t, mask=token_valid, other=0)
        seq_end = tl.load(TOKEN_SEQ_END + offs_t, mask=token_valid, other=0)
        seq_slot = tl.load(TOKEN_SEQ_SLOT + offs_t, mask=token_valid, other=0)
        acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
        for j in range(KERNEL_SIZE):
            source_t = offs_t - (KERNEL_SIZE - 1 - j)
            live = (source_t >= seq_start) & token_valid
            xv = tl.load(
                X + source_t[:, None] * stride_xt + offs_c[None, :] * stride_xc,
                mask=live[:, None] & channel_valid[None, :], other=0.0,
            ).to(tl.float32)
            w = tl.load(
                WEIGHT + offs_c * stride_wc + j * stride_wk,
                mask=channel_valid, other=0.0,
            ).to(tl.float32)
            acc += xv * w[None, :]
        tl.store(
            OUT + offs_t[:, None] * stride_ot + offs_c[None, :] * stride_oc,
            acc, mask=token_valid[:, None] & channel_valid[None, :],
        )

        # The program containing a sequence's last token also writes its exact
        # ks-1 recurrent state into the existing slot-indexed state table.
        is_last = token_valid & (offs_t == seq_end - 1)
        for j in range(KERNEL_SIZE - 1):
            source_t = seq_end - (KERNEL_SIZE - 1) + j
            live = is_last & (source_t >= seq_start)
            value = tl.load(
                X + source_t[:, None] * stride_xt + offs_c[None, :] * stride_xc,
                mask=live[:, None] & channel_valid[None, :], other=0.0,
            )
            tl.store(
                STATE + seq_slot[:, None] * stride_ss + offs_c[None, :] * stride_sc
                + j * stride_sk,
                value, mask=is_last[:, None] & channel_valid[None, :],
            )

    @triton.jit
    def _linear_head_rms_kernel(
        X, W, BIAS, OUT,
        stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
        M, K: tl.constexpr, HEAD_DIM: tl.constexpr,
        HAS_BIAS: tl.constexpr, GATED: tl.constexpr, EPS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        width: tl.constexpr = HEAD_DIM * (2 if GATED else 1)
        offs_n = tl.arange(0, width)
        n_base = head * width
        acc = tl.zeros((BLOCK_M, width), tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            xv = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
            )
            wv = tl.load(
                W + (n_base + offs_n[:, None]) * stride_wn + offs_k[None, :] * stride_wk,
                mask=offs_k[None, :] < K, other=0.0,
            )
            acc += tl.dot(xv, tl.trans(wv), input_precision="ieee")
        if HAS_BIAS:
            acc += tl.load(BIAS + n_base + offs_n)[None, :]
        q_cols = offs_n < HEAD_DIM
        rstd = tl.rsqrt(tl.sum(tl.where(q_cols[None, :], acc * acc, 0.0), axis=1)
                         / HEAD_DIM + EPS)
        result = tl.where(q_cols[None, :], acc * rstd[:, None], acc)
        tl.store(
            OUT + offs_m[:, None] * stride_om + (n_base + offs_n[None, :]) * stride_on,
            result, mask=offs_m[:, None] < M,
        )

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


@torch.no_grad()
def packed_causal_conv1d(
    x, weight, state_table, token_seq_starts, token_seq_ends, token_seq_slots,
):
    """Fresh causal depthwise convolution over packed variable-length sequences."""
    if not HAS_TRITON or not x.is_cuda:
        raise RuntimeError("packed_causal_conv1d requires CUDA and Triton")
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError("expected x [tokens, channels], weight [channels, kernel]")
    kernel_size = weight.shape[1]
    if kernel_size not in (3, 4):
        raise ValueError("packed convolution supports ATMA kernel sizes 3 and 4")
    out = torch.empty_like(x)
    _packed_causal_conv1d_kernel[(triton.cdiv(x.shape[0], 16), triton.cdiv(x.shape[1], 256))](
        x, weight, out, state_table,
        token_seq_starts, token_seq_ends, token_seq_slots,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), state_table.stride(0),
        state_table.stride(1), state_table.stride(2), x.shape[0], x.shape[1],
        KERNEL_SIZE=kernel_size, BLOCK_T=16, BLOCK_C=256,
        num_warps=4,
    )
    return out


@torch.no_grad()
def linear_head_rms(x, weight, bias, *, num_heads, head_dim=128, gated=False, eps=1e-6):
    """Narrow fused projection and affine-free per-head RMSNorm prototype.

    With ``gated=True``, each head stores Q followed by its unchanged gate and
    only the Q half is normalized. cuBLAS plus standalone RMSNorm remains the
    production fallback until a measured shape clears the plan's speed gates.
    """
    if not HAS_TRITON or not x.is_cuda:
        raise RuntimeError("linear_head_rms requires CUDA and Triton")
    width = head_dim * (2 if gated else 1)
    if x.ndim != 2 or weight.shape != (num_heads * width, x.shape[1]):
        raise ValueError("incompatible x/weight shape for head projection")
    if head_dim != 128:
        raise ValueError("the prototype is shape-dispatched only for head_dim=128")
    x, weight = x.contiguous(), weight.contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    block_m = 16 if x.shape[0] > 1 else 1
    _linear_head_rms_kernel[(triton.cdiv(x.shape[0], block_m), num_heads)](
        x, weight, bias if bias is not None else x, out,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), x.shape[0],
        K=x.shape[1], HEAD_DIM=head_dim, HAS_BIAS=bias is not None,
        GATED=gated, EPS=eps, BLOCK_M=block_m, BLOCK_K=32,
        num_warps=4, num_stages=2,
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
