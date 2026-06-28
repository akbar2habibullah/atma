from __future__ import annotations

import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo


def conv_step_uop(out: UOp, new_state: UOp, x: UOp, prev: UOp, weight: UOp) -> UOp:
    """Tinygrad UOp DSL kernel for one-token depthwise causal convolution."""
    channel = UOp.range(out.shape[0], 0, AxisType.GLOBAL)
    kernel = weight.shape[2]
    acc = x[channel] * weight[channel, 0, kernel - 1]
    for idx in range(kernel - 1):
        acc = acc + prev[channel, idx] * weight[channel, 0, idx]

    stores = [out[channel].store(acc)]
    for idx in range(kernel - 2):
        stores.append(new_state[channel, idx].store(prev[channel, idx + 1]))
    stores.append(new_state[channel, kernel - 2].store(x[channel]))
    return UOp.sink(*(store.end(channel) for store in stores), arg=KernelInfo("edge_conv_step"))


def causal_conv1d_decode_step(
    x: Tensor,
    prev: Tensor,
    weight: Tensor,
    *,
    out: Tensor | None = None,
    next_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the custom one-token conv kernel behind a tensor-level API."""
    channels = x.shape[-1] if len(x.shape) == 3 else x.shape[0]
    kernel_m1 = prev.shape[-1]
    flat_x = x.reshape(channels)
    flat_prev = prev.reshape(channels, kernel_m1).cast(x.dtype)
    out = out if out is not None else Tensor.empty(channels, device=x.device, dtype=x.dtype)
    next_state = next_state if next_state is not None else Tensor.empty(channels, kernel_m1, device=x.device, dtype=x.dtype)
    y, state = Tensor.custom_kernel(out, next_state, flat_x, flat_prev, weight.cast(x.dtype), fxn=conv_step_uop)[:2]
    return y, state


def gdn_step_uop(out: UOp, new_state: UOp, q: UOp, k: UOp, v: UOp, gamma: UOp, beta: UOp, state: UOp) -> UOp:
    """One-token gated-delta memory update/read for one flattened (head, row)."""
    idx = UOp.range(out.shape[0], 0, AxisType.GLOBAL)
    head_dim = q.shape[1]
    head = idx // head_dim
    row = idx % head_dim

    g = gamma[head]
    b = beta[head]
    pred = state[head, row, 0] * g * k[head, 0]
    for col in range(1, head_dim):
        pred = pred + state[head, row, col] * g * k[head, col]

    update = b * (v[head, row] - pred)
    read = (state[head, row, 0] * g + update * k[head, 0]) * q[head, 0]
    stores = [new_state[head, row, 0].store(state[head, row, 0] * g + update * k[head, 0])]
    for col in range(1, head_dim):
        next_value = state[head, row, col] * g + update * k[head, col]
        stores.append(new_state[head, row, col].store(next_value))
        read = read + next_value * q[head, col]
    stores.append(out[head, row].store(read))
    return UOp.sink(*(store.end(idx) for store in stores), arg=KernelInfo("edge_gdn_step"))


def gdn_decode_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gamma: Tensor,
    beta: Tensor,
    state: Tensor,
    *,
    out: Tensor | None = None,
    next_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run one fused Titans gated-delta read/update step."""
    heads, head_dim = q.shape[-2], q.shape[-1]
    q_flat = q.reshape(heads, head_dim).float()
    k_flat = k.reshape(heads, head_dim).float()
    v_flat = v.reshape(heads, head_dim).float()
    gamma_flat = gamma.reshape(heads).float()
    beta_flat = beta.reshape(heads).float()
    state_flat = state.reshape(heads, head_dim, head_dim).float()
    out = out if out is not None else Tensor.empty(heads, head_dim, device=q.device, dtype=q_flat.dtype)
    next_state = next_state if next_state is not None else Tensor.empty(heads, head_dim, head_dim, device=q.device, dtype=state_flat.dtype)
    read, new_state = Tensor.custom_kernel(out, next_state, q_flat, k_flat, v_flat, gamma_flat, beta_flat, state_flat, fxn=gdn_step_uop)[:2]
    return read, new_state


def polar_decode_uop(
    content: UOp,
    mag: UOp,
    q: UOp,
    k: UOp,
    v: UOp,
    pos_next: UOp,
    start_pos: UOp,
    n_keys: UOp,
    v_null: UOp,
    null_base: UOp,
    null_slope_raw: UOp,
    len_gain_raw: UOp,
    mag_beta_raw: UOp,
) -> UOp:
    """One-token polar attention reduce for one head."""
    head = UOp.range(content.shape[0], 0, AxisType.GLOBAL)
    max_context, head_dim = k.shape[1], q.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    pos_i = pos_next[0]
    start_i = start_pos[0]
    n = n_keys[0].maximum(1.0)
    temp = 1.0 + len_gain_raw[head].softplus() * n.log()
    null_logit = (null_base[head] + null_slope_raw[head].softplus() * (n + 1.0).log().sqrt()) * temp

    max_logit = null_logit
    logits = []
    for key_idx in range(max_context):
        score = q[head, 0] * k[head, key_idx, 0]
        for dim in range(1, head_dim):
            score = score + q[head, dim] * k[head, key_idx, dim]
        logit = score * scale * temp
        valid = (key_idx < pos_i) & (start_i <= key_idx)
        max_logit = valid.where(logit.maximum(max_logit), max_logit)
        logits.append((logit, valid))

    exp_null = (null_logit - max_logit).exp()
    sum_real = exp_null.const_like(0.0)
    sum_sq_real = exp_null.const_like(0.0)
    weighted = [exp_null * v_null[head, dim] for dim in range(head_dim)]
    for key_idx, (logit, valid) in enumerate(logits):
        weight = valid.where((logit - max_logit).exp(), exp_null.const_like(0.0))
        sum_real = sum_real + weight
        sum_sq_real = sum_sq_real + weight * weight
        for dim in range(head_dim):
            weighted[dim] = weighted[dim] + weight * v[head, key_idx, dim]

    total = sum_real + exp_null
    norm_sq = weighted[0] * weighted[0]
    for dim in range(1, head_dim):
        norm_sq = norm_sq + weighted[dim] * weighted[dim]
    inv_norm = norm_sq.sqrt().maximum(1e-6).reciprocal()

    denom_real = sum_real.maximum(1e-6)
    n_eff = denom_real * denom_real / sum_sq_real.maximum(1e-6)
    m_eff = n_eff * (sum_real / total.maximum(1e-6))
    mag_value = (mag_beta_raw[head].softplus() * (1.0 + m_eff).log()).tanh()

    stores = [mag[head].store(mag_value)]
    for dim in range(head_dim):
        stores.append(content[head, dim].store(weighted[dim] * inv_norm))
    return UOp.sink(*(store.end(head) for store in stores), arg=KernelInfo("edge_polar_decode"))


def polar_decode_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_next: Tensor,
    start_pos: Tensor,
    n_keys: Tensor,
    *,
    v_null: Tensor,
    null_base: Tensor,
    null_slope_raw: Tensor,
    len_gain_raw: Tensor,
    mag_beta_raw: Tensor,
    content: Tensor | None = None,
    mag: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the custom one-token polar attention reducer behind a tensor-level API."""
    heads = q.shape[1] if len(q.shape) == 4 else q.shape[-2]
    head_dim = q.shape[-1]
    max_context = k.shape[2] if len(k.shape) == 4 else k.shape[-2]
    q_flat = q.reshape(heads, head_dim).float()
    k_flat = k.reshape(heads, max_context, head_dim).float()
    v_flat = v.reshape(heads, max_context, head_dim).float()
    content = content if content is not None else Tensor.empty(heads, head_dim, device=q.device, dtype=dtypes.float32)
    mag = mag if mag is not None else Tensor.empty(heads, device=q.device, dtype=dtypes.float32)
    c, m = Tensor.custom_kernel(
        content,
        mag,
        q_flat,
        k_flat,
        v_flat,
        pos_next.reshape(1).cast(dtypes.int32),
        start_pos.reshape(1).cast(dtypes.int32),
        n_keys.reshape(1).float(),
        v_null.float(),
        null_base.float(),
        null_slope_raw.float(),
        len_gain_raw.float(),
        mag_beta_raw.float(),
        fxn=polar_decode_uop,
    )[:2]
    return c, m


def polar_prefill_uop(
    content: UOp,
    mag: UOp,
    q: UOp,
    k: UOp,
    v: UOp,
    window_size: UOp,
    v_null: UOp,
    null_base: UOp,
    null_slope_raw: UOp,
    len_gain_raw: UOp,
    mag_beta_raw: UOp,
) -> UOp:
    """Causal polar attention prefill without materializing the score matrix."""
    idx = UOp.range(content.shape[0], 0, AxisType.GLOBAL)
    tokens, head_dim = q.shape[1], q.shape[2]
    head = idx // tokens
    tok = idx % tokens
    scale = 1.0 / math.sqrt(head_dim)
    window = window_size[0]
    n_abs = tok + 1
    start_i = (n_abs - window).maximum(0)
    n = n_abs.minimum(window).float().maximum(1.0)
    temp = 1.0 + len_gain_raw[head].softplus() * n.log()
    null_logit = (null_base[head] + null_slope_raw[head].softplus() * (n + 1.0).log().sqrt()) * temp

    max_logit = null_logit
    logits = []
    for key_idx in range(tokens):
        score = q[head, tok, 0] * k[head, key_idx, 0]
        for dim in range(1, head_dim):
            score = score + q[head, tok, dim] * k[head, key_idx, dim]
        logit = score * scale * temp
        valid = (key_idx <= tok) & (start_i <= key_idx)
        max_logit = valid.where(logit.maximum(max_logit), max_logit)
        logits.append((logit, valid))

    exp_null = (null_logit - max_logit).exp()
    sum_real = exp_null.const_like(0.0)
    sum_sq_real = exp_null.const_like(0.0)
    weighted = [exp_null * v_null[head, dim] for dim in range(head_dim)]
    for key_idx, (logit, valid) in enumerate(logits):
        weight = valid.where((logit - max_logit).exp(), exp_null.const_like(0.0))
        sum_real = sum_real + weight
        sum_sq_real = sum_sq_real + weight * weight
        for dim in range(head_dim):
            weighted[dim] = weighted[dim] + weight * v[head, key_idx, dim]

    total = sum_real + exp_null
    norm_sq = weighted[0] * weighted[0]
    for dim in range(1, head_dim):
        norm_sq = norm_sq + weighted[dim] * weighted[dim]
    inv_norm = norm_sq.sqrt().maximum(1e-6).reciprocal()

    denom_real = sum_real.maximum(1e-6)
    n_eff = denom_real * denom_real / sum_sq_real.maximum(1e-6)
    m_eff = n_eff * (sum_real / total.maximum(1e-6))
    mag_value = (mag_beta_raw[head].softplus() * (1.0 + m_eff).log()).tanh()

    stores = [mag[idx].store(mag_value)]
    for dim in range(head_dim):
        stores.append(content[idx, dim].store(weighted[dim] * inv_norm))
    return UOp.sink(*(store.end(idx) for store in stores), arg=KernelInfo("edge_polar_prefill"))


def polar_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    window_size: int,
    v_null: Tensor,
    null_base: Tensor,
    null_slope_raw: Tensor,
    len_gain_raw: Tensor,
    mag_beta_raw: Tensor,
    content: Tensor | None = None,
    mag: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run flash-style causal polar attention prefill for standalone profiling."""
    heads, tokens, head_dim = q.shape[-3], q.shape[-2], q.shape[-1]
    q_flat = q.reshape(heads, tokens, head_dim).float()
    k_flat = k.reshape(heads, tokens, head_dim).float()
    v_flat = v.reshape(heads, tokens, head_dim).float()
    content = content.reshape(heads * tokens, head_dim) if content is not None else Tensor.empty(heads * tokens, head_dim, device=q.device, dtype=dtypes.float32)
    mag = mag.reshape(heads * tokens) if mag is not None else Tensor.empty(heads * tokens, device=q.device, dtype=dtypes.float32)
    c, m = Tensor.custom_kernel(
        content,
        mag,
        q_flat,
        k_flat,
        v_flat,
        Tensor([window_size], device=q.device, dtype=dtypes.int32),
        v_null.float(),
        null_base.float(),
        null_slope_raw.float(),
        len_gain_raw.float(),
        mag_beta_raw.float(),
        fxn=polar_prefill_uop,
    )[:2]
    return c.reshape(heads, tokens, head_dim), m.reshape(heads, tokens)


def gdn_prefill_uop(out: UOp, new_state: UOp, q: UOp, k: UOp, v: UOp, gamma: UOp, beta: UOp, state: UOp) -> UOp:
    """Causal gated-delta prefill scan for one flattened (head, value row)."""
    idx = UOp.range(out.shape[0], 0, AxisType.GLOBAL)
    tokens, head_dim = q.shape[1], q.shape[2]
    head = idx // head_dim
    row = idx % head_dim
    cur = [state[head, row, col] for col in range(head_dim)]
    stores = []
    for tok in range(tokens):
        g = gamma[head, tok]
        b = beta[head, tok]
        pred = cur[0] * g * k[head, tok, 0]
        for col in range(1, head_dim):
            pred = pred + cur[col] * g * k[head, tok, col]
        update = b * (v[head, tok, row] - pred)
        read = (cur[0] * g + update * k[head, tok, 0]) * q[head, tok, 0]
        cur[0] = cur[0] * g + update * k[head, tok, 0]
        for col in range(1, head_dim):
            next_value = cur[col] * g + update * k[head, tok, col]
            read = read + next_value * q[head, tok, col]
            cur[col] = next_value
        stores.append(out[idx, tok].store(read))
    for col in range(head_dim):
        stores.append(new_state[idx, col].store(cur[col]))
    return UOp.sink(*(store.end(idx) for store in stores), arg=KernelInfo("edge_gdn_prefill"))


def gdn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gamma: Tensor,
    beta: Tensor,
    state: Tensor,
    *,
    out: Tensor | None = None,
    next_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run flash-style Titans gated-delta prefill for standalone profiling."""
    heads, tokens, head_dim = q.shape[-3], q.shape[-2], q.shape[-1]
    q_flat = q.reshape(heads, tokens, head_dim).float()
    k_flat = k.reshape(heads, tokens, head_dim).float()
    v_flat = v.reshape(heads, tokens, head_dim).float()
    gamma_flat = gamma.reshape(heads, tokens).float()
    beta_flat = beta.reshape(heads, tokens).float()
    state_flat = state.reshape(heads, head_dim, head_dim).float()
    out = out.reshape(heads * head_dim, tokens) if out is not None else Tensor.empty(heads * head_dim, tokens, device=q.device, dtype=dtypes.float32)
    next_state = next_state.reshape(heads * head_dim, head_dim) if next_state is not None else Tensor.empty(heads * head_dim, head_dim, device=q.device, dtype=dtypes.float32)
    read, final_state = Tensor.custom_kernel(out, next_state, q_flat, k_flat, v_flat, gamma_flat, beta_flat, state_flat, fxn=gdn_prefill_uop)[:2]
    return read.reshape(heads, head_dim, tokens).transpose(1, 2), final_state.reshape(heads, head_dim, head_dim)


def gdn_prefill_chunked(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gamma: Tensor,
    beta: Tensor,
    state: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Run GDN prefill as multiple token chunks to bound custom-kernel source size."""
    tokens = q.shape[-2]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    reads: list[Tensor] = []
    cur = state
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        read, cur = gdn_prefill(
            q[:, start:end, :],
            k[:, start:end, :],
            v[:, start:end, :],
            gamma[:, start:end],
            beta[:, start:end],
            cur,
        )
        reads.append(read)
    return (reads[0] if len(reads) == 1 else reads[0].cat(*reads[1:], dim=1)), cur
