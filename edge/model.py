from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.nn import state as tg_state

from edge.kernels import causal_conv1d_decode_step, gdn_decode_step, polar_decode_step
from model.config import AtmaConfig


_LEN_GAIN_INIT = -1.0
_NULL_SLOPE_INIT = 0.5
_NULL_BASE_INIT = 2.0
_MAG_BETA_INIT = -1.5
_TORCH_FP32_EPS = 1.1920928955078125e-7


def _to_tiny(value: Any, device: str | None = None, dtype=None) -> Tensor:
    if isinstance(value, Tensor):
        out = value
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        out = Tensor(value.detach().cpu().numpy(), device=device)
    else:
        out = Tensor(np.asarray(value), device=device)
    return out.cast(dtype) if dtype is not None and not dtypes.is_int(out.dtype) else out


def _get_child(obj: Any, name: str) -> Any:
    if isinstance(obj, list):
        return obj[int(name)]
    return getattr(obj, name)


def _set_path(obj: Any, path: str, value: Tensor) -> None:
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    setattr(parent, parts[-1], value)


def _rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = _TORCH_FP32_EPS) -> Tensor:
    y = (x.float() * (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype)
    return y if weight is None else y * weight


def _normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x.float() / x.float().square().sum(-1, keepdim=True).sqrt().maximum(eps)


class EdgeLinear(nn.Linear):
    pass


class EdgeRMSNorm(nn.RMSNorm):
    pass


class EdgeDepthwiseConv1d:
    def __init__(self, channels: int, kernel_size: int, device: str | None = None, dtype=dtypes.float32):
        bound = 1 / max(1, kernel_size) ** 0.5
        self.weight = Tensor.uniform(channels, 1, kernel_size, low=-bound, high=bound, device=device, dtype=dtype)


class EdgeMLP:
    def __init__(self, dim: int):
        hidden = 4 * dim
        self.fc = EdgeLinear(dim, hidden * 2)
        self.proj = EdgeLinear(hidden, dim)

    def __call__(self, x: Tensor) -> Tensor:
        x_val, gate = self.fc(x).chunk(2, dim=-1)
        return self.proj(gate * x_val.relu().square())


@dataclass
class EdgeState:
    """Mutable tinygrad state for one active prompt/session."""

    conv: dict[str, Tensor] = field(default_factory=dict)
    k_cache: dict[int, Tensor] = field(default_factory=dict)
    v_cache: dict[int, Tensor] = field(default_factory=dict)
    mem: dict[int, Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        self.conv.clear()
        self.k_cache.clear()
        self.v_cache.clear()
        self.mem.clear()

    def tensors(self) -> list[Tensor]:
        return [*self.conv.values(), *self.k_cache.values(), *self.v_cache.values(), *self.mem.values()]

    def realize(self, *extra: Tensor) -> None:
        tensors = [*extra, *self.tensors()]
        if tensors:
            Tensor.realize(*tensors)


@dataclass
class EdgeStaticState:
    """Fixed-shape decode state for TinyJit one-token replay."""

    max_context: int
    conv: dict[str, Tensor] = field(default_factory=dict)
    conv_next: dict[str, Tensor] = field(default_factory=dict)
    conv_out: dict[str, Tensor] = field(default_factory=dict)
    k_cache: dict[int, Tensor] = field(default_factory=dict)
    v_cache: dict[int, Tensor] = field(default_factory=dict)
    mem: dict[int, Tensor] = field(default_factory=dict)
    mem_next: dict[int, Tensor] = field(default_factory=dict)
    mem_read: dict[int, Tensor] = field(default_factory=dict)
    polar_content: dict[int, Tensor] = field(default_factory=dict)
    polar_mag: dict[int, Tensor] = field(default_factory=dict)

    def tensors(self) -> list[Tensor]:
        return [
            *self.conv.values(),
            *self.conv_next.values(),
            *self.conv_out.values(),
            *self.k_cache.values(),
            *self.v_cache.values(),
            *self.mem.values(),
            *self.mem_next.values(),
            *self.mem_read.values(),
            *self.polar_content.values(),
            *self.polar_mag.values(),
        ]

    def realize(self, *extra: Tensor) -> None:
        tensors = [*extra, *self.tensors()]
        if tensors:
            Tensor.realize(*tensors)

    @property
    def cache_len(self) -> int:
        if not self.k_cache:
            return 0
        return next(iter(self.k_cache.values())).shape[1]


def _causal_conv1d_stateful(key: str, x: Tensor, conv: EdgeDepthwiseConv1d, state: EdgeState) -> Tensor:
    batch, tokens, channels = x.shape
    if batch != 1:
        raise ValueError("edge tinygrad runtime currently supports batch size 1 per EdgeState")

    kernel = conv.weight.shape[-1]
    x_stream = x.transpose(1, 2)  # (B, C, T)
    if kernel == 1:
        return (x_stream * conv.weight[:, 0, 0].reshape(1, channels, 1)).transpose(1, 2)

    prev = state.conv.get(key)
    if prev is None:
        prev = Tensor.zeros(batch, channels, kernel - 1, device=x.device, dtype=x.dtype)
    if tokens == 1:
        y, next_state = causal_conv1d_decode_step(x, prev, conv.weight)
        state.conv[key] = next_state.reshape(batch, channels, kernel - 1).detach()
        return y.reshape(batch, tokens, channels)

    padded = prev.cast(x.dtype).cat(x_stream, dim=2)
    out = padded[:, :, :tokens] * conv.weight[:, 0, 0].reshape(1, channels, 1).cast(x.dtype)
    for i in range(1, kernel):
        out = out + padded[:, :, i:i + tokens] * conv.weight[:, 0, i].reshape(1, channels, 1).cast(x.dtype)
    state.conv[key] = padded[:, :, -(kernel - 1):].detach()
    return out.transpose(1, 2)


def _causal_conv1d_static_step(key: str, x: Tensor, conv: EdgeDepthwiseConv1d, state: EdgeStaticState) -> Tensor:
    batch, tokens, channels = x.shape
    if batch != 1 or tokens != 1:
        raise ValueError("static decode conv supports exactly one token")
    kernel = conv.weight.shape[-1]
    x_stream = x.transpose(1, 2)
    if kernel == 1:
        return (x_stream * conv.weight[:, 0, 0].reshape(1, channels, 1)).transpose(1, 2)

    y, next_state = causal_conv1d_decode_step(
        x,
        state.conv[key],
        conv.weight,
        out=state.conv_out[key],
        next_state=state.conv_next[key],
    )[:2]
    Tensor.realize(y, next_state)
    state.conv[key].assign(next_state.reshape(batch, channels, kernel - 1)).realize()
    return y.reshape(batch, tokens, channels)


def _polar_temp_null(n_keys: Tensor, len_gain_raw: Tensor, null_base: Tensor, null_slope_raw: Tensor) -> tuple[Tensor, Tensor]:
    heads = len_gain_raw.shape[0]
    n = n_keys.maximum(1.0)
    temp = 1.0 + len_gain_raw.softplus().reshape(1, heads, 1, 1) * n.log().reshape(1, 1, -1, 1)
    null = null_base.reshape(1, heads, 1, 1) + null_slope_raw.softplus().reshape(1, heads, 1, 1) * (n + 1.0).log().sqrt().reshape(1, 1, -1, 1)
    return temp, null


def _polar_reduce(
    sigma: Tensor,
    v: Tensor,
    n_keys: Tensor,
    *,
    v_null: Tensor,
    null_base: Tensor,
    null_slope_raw: Tensor,
    len_gain_raw: Tensor,
    mag_beta_raw: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    out_dtype = v.dtype
    sigma = sigma.float()
    v = v.float()
    batch, heads, t_query, _ = sigma.shape
    head_dim = v.shape[-1]

    temp, null = _polar_temp_null(n_keys.float(), len_gain_raw.float(), null_base.float(), null_slope_raw.float())
    real = sigma * temp
    logits = real.cat(null.expand(batch, heads, t_query, 1) * temp, dim=-1)
    weights = logits.softmax(-1)
    w_real = weights[..., :-1]
    w_null = weights[..., -1:]

    mixed = (w_real @ v) + w_null * v_null.float().reshape(1, heads, 1, head_dim)
    content = _normalize(mixed, eps=eps)

    denom = w_real.sum(-1, keepdim=True).maximum(eps)
    w_hat = w_real / denom
    n_eff = 1.0 / w_hat.square().sum(-1).maximum(eps)
    m_eff = n_eff * (1.0 - w_null.squeeze(-1))
    mag = (mag_beta_raw.float().softplus().reshape(1, heads, 1) * (1.0 + m_eff).log()).tanh()
    return content.cast(out_dtype), mag.cast(out_dtype)


def _gated_delta_sequential(q: Tensor, k: Tensor, v: Tensor, gamma: Tensor, beta: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    batch, heads, tokens, head_dim = q.shape
    reads = []
    s = state.float()
    for i in range(tokens):
        qi = q[:, :, i, :].float()
        ki = k[:, :, i, :].float()
        vi = v[:, :, i, :].float()
        gi = gamma[:, :, i].float().reshape(batch, heads, 1, 1)
        bi = beta[:, :, i].float().reshape(batch, heads, 1)
        decayed = s * gi
        pred = (decayed * ki.reshape(batch, heads, 1, head_dim)).sum(-1)
        update = bi * (vi - pred)
        s = decayed + update.reshape(batch, heads, head_dim, 1) * ki.reshape(batch, heads, 1, head_dim)
        reads.append((s * qi.reshape(batch, heads, 1, head_dim)).sum(-1))
    return Tensor.stack(*reads, dim=2), s


class EdgeTitansMemory:
    def __init__(self, dim: int, heads: int, head_dim: int, chunk: int, gamma_bias: float, beta_bias: float):
        self.H, self.dk, self.chunk = heads, head_dim, chunk
        self.gamma_bias, self.beta_bias = gamma_bias, beta_bias
        self.w_gamma = EdgeLinear(dim, heads)
        self.w_beta = EdgeLinear(dim, heads)
        self.gate = EdgeLinear(dim, heads * head_dim)
        self.proj = EdgeLinear(heads * head_dim, dim)
        self.proj.weight = Tensor.zeros(*self.proj.weight.shape)
        if self.proj.bias is not None:
            self.proj.bias = Tensor.zeros(*self.proj.bias.shape)


class EdgeLFM2Conv:
    def __init__(self, layer_idx: int, dim: int, kernel_size: int):
        self.layer_idx = layer_idx
        self.hidden_size = dim
        self.kernel_size = kernel_size
        self.in_proj = EdgeLinear(dim, 3 * dim)
        self.conv = EdgeDepthwiseConv1d(dim, kernel_size)
        self.out_proj = EdgeLinear(dim, dim)

    def __call__(self, x: Tensor, state: EdgeState) -> Tensor:
        b_gate, carry, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = b_gate * x_proj
        x_conv = _causal_conv1d_stateful(f"conv_{self.layer_idx}_gated", x_gated, self.conv, state)
        return self.out_proj(carry * x_conv)

    def decode_static(self, x: Tensor, state: EdgeStaticState) -> Tensor:
        b_gate, carry, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = b_gate * x_proj
        x_conv = _causal_conv1d_static_step(f"conv_{self.layer_idx}_gated", x_gated, self.conv, state)
        return self.out_proj(carry * x_conv)


class EdgePolarAttention:
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        head_dim: int,
        num_kv_heads: int,
        kernel_size: int,
        window: int | None,
        mem_enabled: bool,
        mem_chunk: int,
        mem_gamma_bias: float,
        mem_beta_bias: float,
    ):
        self.layer_idx = layer_idx
        self.num_heads = dim // head_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hdim = self.num_heads * self.head_dim
        self.kv_hdim = self.num_kv_heads * self.head_dim
        self.kernel_size = kernel_size
        self.window = window

        self.q = EdgeLinear(dim, self.hdim * 2)
        self.k = EdgeLinear(dim, self.kv_hdim)
        self.v = EdgeLinear(dim, self.kv_hdim)
        self.canon_q = EdgeDepthwiseConv1d(self.hdim, kernel_size)
        self.canon_k = EdgeDepthwiseConv1d(self.kv_hdim, kernel_size)
        self.canon_v = EdgeDepthwiseConv1d(self.kv_hdim, kernel_size)
        self.proj = EdgeLinear(self.hdim, dim)

        self.mu_proj = EdgeLinear(self.num_heads, dim)
        self.v_null = Tensor.zeros(self.num_heads, head_dim)
        self.null_base = Tensor.full((self.num_heads,), _NULL_BASE_INIT)
        self.null_slope_raw = Tensor.full((self.num_heads,), _NULL_SLOPE_INIT)
        self.len_gain_raw = Tensor.full((self.num_heads,), _LEN_GAIN_INIT)
        self.mag_beta_raw = Tensor.full((self.num_heads,), _MAG_BETA_INIT)
        self.mem = EdgeTitansMemory(dim, self.num_heads, head_dim, mem_chunk, mem_gamma_bias, mem_beta_bias) if mem_enabled else None

    def _memory(self, x: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor, state: EdgeState) -> Tensor:
        batch, tokens = x.shape[0], x.shape[1]
        heads, head_dim = self.num_heads, self.head_dim
        old_state = state.mem.get(self.layer_idx)
        if old_state is None:
            old_state = Tensor.zeros(batch, heads, head_dim, head_dim, device=x.device, dtype=dtypes.float32)

        assert self.mem is not None
        g_logit = self.mem.w_gamma(x).float() + self.mem.gamma_bias
        b_logit = self.mem.w_beta(x).float() + self.mem.beta_bias
        gamma = g_logit.sigmoid().transpose(1, 2)
        beta = b_logit.sigmoid().transpose(1, 2)
        read, new_state = _gated_delta_sequential(_normalize(q_t), _normalize(k_t), v_t.float(), gamma, beta, old_state)
        state.mem[self.layer_idx] = new_state.detach()

        read = _rms_norm(read.transpose(1, 2))
        read_flat = read.reshape(batch, tokens, heads * head_dim).cast(x.dtype)
        return self.mem.proj(read_flat * self.mem.gate(x).sigmoid())

    def _memory_static(self, x: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor, state: EdgeStaticState) -> Tensor:
        batch, tokens = x.shape[0], x.shape[1]
        heads, head_dim = self.num_heads, self.head_dim
        if batch != 1 or tokens != 1:
            raise ValueError("static memory decode supports exactly one token")

        assert self.mem is not None
        g_logit = self.mem.w_gamma(x).float() + self.mem.gamma_bias
        b_logit = self.mem.w_beta(x).float() + self.mem.beta_bias
        gamma = g_logit.sigmoid().reshape(heads)
        beta = b_logit.sigmoid().reshape(heads)
        read, new_state = gdn_decode_step(
            _normalize(q_t).reshape(heads, head_dim),
            _normalize(k_t).reshape(heads, head_dim),
            v_t.float().reshape(heads, head_dim),
            gamma,
            beta,
            state.mem[self.layer_idx],
            out=state.mem_read[self.layer_idx],
            next_state=state.mem_next[self.layer_idx],
        )
        Tensor.realize(read, new_state)
        state.mem[self.layer_idx].assign(new_state.reshape(1, heads, head_dim, head_dim)).realize()

        read = _rms_norm(read.reshape(batch, heads, tokens, head_dim).transpose(1, 2))
        read_flat = read.reshape(batch, tokens, heads * head_dim).cast(x.dtype)
        return self.mem.proj(read_flat * self.mem.gate(x).sigmoid())

    def __call__(self, x: Tensor, state: EdgeState) -> Tensor:
        batch, tokens, _ = x.shape
        if batch != 1:
            raise ValueError("edge tinygrad runtime currently supports batch size 1 per EdgeState")

        heads, head_dim = self.num_heads, self.head_dim
        q_gate = self.q(x).reshape(batch, tokens, heads, head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k(x).reshape(batch, tokens, self.num_kv_heads, head_dim)
        v = self.v(x).reshape(batch, tokens, self.num_kv_heads, head_dim)

        q = _rms_norm(q)
        k = _rms_norm(k)

        q_flat = q.reshape(batch, tokens, -1)
        k_flat = k.reshape(batch, tokens, -1)
        v_flat = v.reshape(batch, tokens, -1)
        q_conv = q_flat + _causal_conv1d_stateful(f"attn_{self.layer_idx}_q", q_flat, self.canon_q, state)
        k_conv = k_flat + _causal_conv1d_stateful(f"attn_{self.layer_idx}_k", k_flat, self.canon_k, state)
        v_conv = v_flat + _causal_conv1d_stateful(f"attn_{self.layer_idx}_v", v_flat, self.canon_v, state)

        q_attn = q_conv.reshape(batch, tokens, heads, head_dim)
        k_new = k_conv.reshape(batch, tokens, self.num_kv_heads, head_dim)
        v_new = v_conv.reshape(batch, tokens, self.num_kv_heads, head_dim)

        k_prev = state.k_cache.get(self.layer_idx)
        v_prev = state.v_cache.get(self.layer_idx)
        prefix_len = 0 if k_prev is None else k_prev.shape[1]
        k_all = k_new if k_prev is None else k_prev.cast(k_new.dtype).cat(k_new, dim=1)
        v_all = v_new if v_prev is None else v_prev.cast(v_new.dtype).cat(v_new, dim=1)
        state.k_cache[self.layer_idx] = k_all.detach()
        state.v_cache[self.layer_idx] = v_all.detach()

        groups = heads // self.num_kv_heads
        q_t = q_attn.transpose(1, 2).contiguous()
        k_t = k_all.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
        v_t = v_all.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()

        t_key = k_t.shape[2]
        n_abs = Tensor.arange(prefix_len + 1, prefix_len + tokens + 1, device=x.device, dtype=dtypes.float32)
        key_idx = Tensor.arange(t_key, device=x.device).reshape(1, -1)
        invalid = key_idx >= n_abs.reshape(-1, 1)
        n_temp = n_abs
        if self.window is not None:
            invalid = invalid | (key_idx < (n_abs.reshape(-1, 1) - self.window))
            n_temp = n_abs.minimum(float(self.window))
        sigma = (q_t.float() @ k_t.float().transpose(-2, -1)) / (head_dim ** 0.5)
        sigma = sigma.masked_fill(invalid.reshape(1, 1, tokens, t_key), float("-inf"))
        c, mag = _polar_reduce(
            sigma,
            v_t,
            n_temp,
            v_null=self.v_null,
            null_base=self.null_base,
            null_slope_raw=self.null_slope_raw,
            len_gain_raw=self.len_gain_raw,
            mag_beta_raw=self.mag_beta_raw,
        )

        c_view = c.reshape(batch, heads, tokens, head_dim)
        mag_view = mag.reshape(batch, heads, tokens)
        c_flat = c_view.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
        out = self.proj(c_flat * gate.reshape(batch, tokens, -1).sigmoid())
        out = out + self.mu_proj(mag_view.transpose(1, 2))
        if self.mem is not None:
            k_mem = k_new.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
            v_mem = v_new.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
            out = out + self._memory(x, q_t, k_mem, v_mem, state)
        return out

    def decode_static(self, x: Tensor, pos, state: EdgeStaticState) -> Tensor:
        batch, tokens, _ = x.shape
        if batch != 1 or tokens != 1:
            raise ValueError("static decode attention supports exactly one token")

        heads, head_dim = self.num_heads, self.head_dim
        q_gate = self.q(x).reshape(batch, tokens, heads, head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k(x).reshape(batch, tokens, self.num_kv_heads, head_dim)
        v = self.v(x).reshape(batch, tokens, self.num_kv_heads, head_dim)

        q = _rms_norm(q)
        k = _rms_norm(k)

        q_flat = q.reshape(batch, tokens, -1)
        k_flat = k.reshape(batch, tokens, -1)
        v_flat = v.reshape(batch, tokens, -1)
        q_conv = q_flat + _causal_conv1d_static_step(f"attn_{self.layer_idx}_q", q_flat, self.canon_q, state)
        k_conv = k_flat + _causal_conv1d_static_step(f"attn_{self.layer_idx}_k", k_flat, self.canon_k, state)
        v_conv = v_flat + _causal_conv1d_static_step(f"attn_{self.layer_idx}_v", v_flat, self.canon_v, state)

        q_attn = q_conv.reshape(batch, tokens, heads, head_dim)
        k_new = k_conv.reshape(batch, tokens, self.num_kv_heads, head_dim)
        v_new = v_conv.reshape(batch, tokens, self.num_kv_heads, head_dim)

        groups = heads // self.num_kv_heads
        max_context = state.max_context
        pos_mask = (Tensor.arange(max_context, device=x.device).reshape(1, max_context, 1, 1) == pos)
        state.k_cache[self.layer_idx].assign(pos_mask.where(k_new.expand(1, max_context, self.num_kv_heads, head_dim), state.k_cache[self.layer_idx])).realize()
        state.v_cache[self.layer_idx].assign(pos_mask.where(v_new.expand(1, max_context, self.num_kv_heads, head_dim), state.v_cache[self.layer_idx])).realize()

        q_t = q_attn.transpose(1, 2).contiguous()
        k_t = state.k_cache[self.layer_idx].repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
        v_t = state.v_cache[self.layer_idx].repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()

        pos_next = pos + 1
        pos_tensor = Tensor.arange(1, device=x.device, dtype=dtypes.int32) + pos_next
        start_tensor = Tensor.zeros(1, device=x.device, dtype=dtypes.int32)
        n_temp = pos_tensor.float()
        if self.window is not None:
            start_tensor = (pos_tensor - self.window).maximum(0)
            n_temp = n_temp.minimum(float(self.window))
        c, mag = polar_decode_step(
            q_t,
            k_t,
            v_t,
            pos_tensor,
            start_tensor,
            n_temp,
            v_null=self.v_null,
            null_base=self.null_base,
            null_slope_raw=self.null_slope_raw,
            len_gain_raw=self.len_gain_raw,
            mag_beta_raw=self.mag_beta_raw,
            content=state.polar_content[self.layer_idx],
            mag=state.polar_mag[self.layer_idx],
        )
        Tensor.realize(c, mag)
        c = c.cast(x.dtype)
        mag = mag.cast(x.dtype)

        c_view = c.reshape(batch, heads, tokens, head_dim)
        mag_view = mag.reshape(batch, heads, tokens)
        c_flat = c_view.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
        out = self.proj(c_flat * gate.reshape(batch, tokens, -1).sigmoid())
        out = out + self.mu_proj(mag_view.transpose(1, 2))
        if self.mem is not None:
            k_mem = k_new.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
            v_mem = v_new.repeat_interleave(groups, dim=2).transpose(1, 2).contiguous()
            out = out + self._memory_static(x, q_t, k_mem, v_mem, state)
        return out


class EdgeDecoderBlock:
    def __init__(self, layer_idx: int, config: AtmaConfig):
        self.attn = (
            EdgePolarAttention(
                layer_idx,
                config.hidden_size,
                config.head_dim,
                config.num_key_value_heads,
                config.attn_kernel_size,
                config.attn_window,
                config.mem_enabled,
                config.mem_chunk,
                config.mem_gamma_bias,
                config.mem_beta_bias,
            )
            if layer_idx % 4 == 2
            else EdgeLFM2Conv(layer_idx, config.hidden_size, config.conv_kernel_size)
        )
        self.mlp = EdgeMLP(config.hidden_size)
        self.norm1 = EdgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = EdgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: Tensor, state: EdgeState) -> Tensor:
        x = x + self.attn(self.norm1(x), state)
        x = x + self.mlp(self.norm2(x))
        return x

    def decode_static(self, x: Tensor, pos, state: EdgeStaticState) -> Tensor:
        attn = self.attn.decode_static(self.norm1(x), pos, state) if isinstance(self.attn, EdgePolarAttention) else self.attn.decode_static(self.norm1(x), state)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class EdgeAtma:
    """tinygrad single-session Atma decoder with an incremental state cache."""

    def __init__(self, config: AtmaConfig):
        self.config = config
        self.device = None
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = [EdgeDecoderBlock(i, config) for i in range(config.num_hidden_layers)]
        self.proj = EdgeLinear(config.hidden_size, config.vocab_size)
        self.norm = EdgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def new_state(self) -> EdgeState:
        return EdgeState()

    def new_static_state(self, max_context: int | None = None) -> EdgeStaticState:
        max_context = max_context or self.config.max_position_embeddings
        out = EdgeStaticState(max_context=max_context)
        device = self.device
        hidden = self.config.hidden_size
        dtype = self.embed.weight.dtype
        for i, block in enumerate(self.blocks):
            if isinstance(block.attn, EdgeLFM2Conv):
                key = f"conv_{i}_gated"
                out.conv[key] = Tensor.zeros(1, hidden, self.config.conv_kernel_size - 1, device=device, dtype=dtype).realize()
                out.conv_next[key] = Tensor.empty(hidden, self.config.conv_kernel_size - 1, device=device, dtype=dtype).realize()
                out.conv_out[key] = Tensor.empty(hidden, device=device, dtype=dtype).realize()
            else:
                attn = block.attn
                for key, channels in ((f"attn_{i}_q", attn.hdim), (f"attn_{i}_k", attn.kv_hdim), (f"attn_{i}_v", attn.kv_hdim)):
                    out.conv[key] = Tensor.zeros(1, channels, self.config.attn_kernel_size - 1, device=device, dtype=dtype).realize()
                    out.conv_next[key] = Tensor.empty(channels, self.config.attn_kernel_size - 1, device=device, dtype=dtype).realize()
                    out.conv_out[key] = Tensor.empty(channels, device=device, dtype=dtype).realize()
                out.k_cache[i] = Tensor.zeros(1, max_context, attn.num_kv_heads, attn.head_dim, device=device, dtype=dtype).realize()
                out.v_cache[i] = Tensor.zeros(1, max_context, attn.num_kv_heads, attn.head_dim, device=device, dtype=dtype).realize()
                out.polar_content[i] = Tensor.empty(attn.num_heads, attn.head_dim, device=device, dtype=dtypes.float32).realize()
                out.polar_mag[i] = Tensor.empty(attn.num_heads, device=device, dtype=dtypes.float32).realize()
                if attn.mem is not None:
                    out.mem[i] = Tensor.zeros(1, attn.num_heads, attn.head_dim, attn.head_dim, device=device, dtype=dtypes.float32).realize()
                    out.mem_next[i] = Tensor.empty(attn.num_heads, attn.head_dim, attn.head_dim, device=device, dtype=dtypes.float32).realize()
                    out.mem_read[i] = Tensor.empty(attn.num_heads, attn.head_dim, device=device, dtype=dtypes.float32).realize()
        return out

    def state_dict(self) -> dict[str, Tensor]:
        return tg_state.get_state_dict(self)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> SimpleNamespace:
        own = self.state_dict()
        missing = [key for key in own if key not in state_dict]
        unexpected = [key for key in state_dict if key not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(f"Error(s) loading state_dict: missing={missing}, unexpected={unexpected}")
        for key, value in state_dict.items():
            if key in own:
                tensor = _to_tiny(value, device=own[key].device, dtype=own[key].dtype)
                if tuple(tensor.shape) != tuple(own[key].shape):
                    raise RuntimeError(f"shape mismatch for {key}: got {tensor.shape}, expected {own[key].shape}")
                _set_path(self, key, tensor)
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

    def to(self, device: str | None = None, dtype=None) -> "EdgeAtma":
        self.device = device or self.device
        for key, value in list(self.state_dict().items()):
            cast_dtype = dtype if dtype is not None and not dtypes.is_int(value.dtype) else value.dtype
            tensor = value.to(device) if device is not None else value
            _set_path(self, key, tensor.cast(cast_dtype).realize())
        return self

    def __call__(self, input_ids: Tensor | list[int], state: EdgeState | None = None) -> Tensor:
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor([input_ids], device=self.device, dtype=dtypes.int32)
        elif input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, input_ids.shape[0])
        if input_ids.shape[0] != 1:
            raise ValueError("edge tinygrad runtime currently supports one sequence per forward")
        state = state or self.new_state()
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, state)
        logits = self.compute_logits(x)
        state.realize(logits)
        return logits

    def decode_static(self, input_ids: Tensor, pos, state: EdgeStaticState) -> Tensor:
        if input_ids.shape != (1, 1):
            raise ValueError("decode_static expects input_ids shape (1, 1)")
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block.decode_static(x, pos, state)
        logits = self.compute_logits(x)
        state.realize(logits)
        return logits

    def compute_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.proj(self.norm(hidden_states)).float()
        return 15.0 * logits * (logits.square() + 225.0).rsqrt()
