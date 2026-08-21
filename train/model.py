import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None
from types import SimpleNamespace

from train.reg import sigreg
from kernels import get_kernel
from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase, polar_reduce, polar_temp_null, polar_attention_online, TitansMemory

try:
    from kernel.polar_triton import (
        polar_attention as polar_attention_triton,
        polar_attention_fwd as polar_attention_triton_fwd,
        HAS_TRITON,
    )
except Exception:
    polar_attention_triton = None
    polar_attention_triton_fwd = None
    HAS_TRITON = False

# Tilde Research Wall Attention (data-dependent per-channel gates lifted into softmax;
# https://github.com/tilde-research/wall-attention-release). Optional GPU/Triton kernel used for
# the faithful memory-efficient path for attn_type="wall"; a pure-PyTorch fallback covers CPU and
# missing-kernel development runs.
_WALL_IMPL = os.environ.get("ATMA_WALL_IMPL", "auto").strip().lower()
if _WALL_IMPL not in {"auto", "local", "upstream"}:
    raise ValueError("ATMA_WALL_IMPL must be one of: auto, local, upstream")

try:
    if _WALL_IMPL == "upstream":
        raise ImportError("ATMA_WALL_IMPL=upstream")
    from kernel.wall import wall_attn as _wall_attn_kernel
    _HAS_WALL = True
except Exception:
    try:
        if _WALL_IMPL == "local":
            raise ImportError("ATMA_WALL_IMPL=local")
        from wall_attn import wall_attn as _wall_attn_kernel
        _HAS_WALL = True
    except Exception:
        _wall_attn_kernel = None
        _HAS_WALL = False

def _wall_kernel_call(q, k, v, g, scale: float, window_size: int):
    return _wall_attn_kernel(q, k, v, g, scale=scale, window_size=(None if window_size == 0 else window_size))


_wall_attn = _wall_kernel_call if _HAS_WALL else None
_WALL_CUSTOM_OP = os.environ.get("ATMA_WALL_CUSTOM_OP", "0") == "1"
if _HAS_WALL and _WALL_CUSTOM_OP:
    try:
        @torch.library.custom_op("atma::wall_attn_fwd", mutates_args=())
        def _wall_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                      scale: float, window_size: int) -> torch.Tensor:
            return _wall_kernel_call(q, k, v, g, scale, window_size)

        @_wall_fwd.register_fake
        def _(q, k, v, g, scale: float, window_size: int):
            return q.new_empty((*q.shape[:-1], v.shape[-1]))

        @torch.library.custom_op("atma::wall_attn_bwd", mutates_args=())
        def _wall_bwd(grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor, g: torch.Tensor, scale: float,
                      window_size: int) -> list[torch.Tensor]:
            with torch.enable_grad():
                ins = [t.detach().requires_grad_(True) for t in (q, k, v, g)]
                o = _wall_kernel_call(*ins, scale, window_size)
                return list(torch.autograd.grad(o, ins, grad_o))

        @_wall_bwd.register_fake
        def _(grad_o, q, k, v, g, scale: float, window_size: int):
            return [torch.empty_like(t) for t in (q, k, v, g)]

        def _wall_setup(ctx, inputs, output):
            q, k, v, g, scale, window_size = inputs
            ctx.save_for_backward(q, k, v, g)
            ctx.scale = scale
            ctx.window_size = window_size

        def _wall_backward(ctx, grad_o):
            grads = _wall_bwd(grad_o, *ctx.saved_tensors, ctx.scale, ctx.window_size)
            return tuple(grads) + (None, None)

        _wall_fwd.register_autograd(_wall_backward, setup_context=_wall_setup)
        _wall_attn = _wall_fwd
    except Exception:
        _wall_attn = _wall_kernel_call

# Polar structural-prior inits (softplus^-1 of validated targets: g~0.3, slope~1, beta~0.2)
_LEN_GAIN_INIT = -1.0
_NULL_SLOPE_INIT = 0.5
_NULL_BASE_INIT = 2.0
_MAG_BETA_INIT = -1.5
_WALL_GATE_BIAS_INIT = 6.0
_WALL_GATE_LOG_MAX = 0.87


def _wall_log_decay(logits: Tensor, *, log_max: float = _WALL_GATE_LOG_MAX) -> Tensor:
    """Map Wall gate logits to bounded natural-log decays for the Triton kernel."""
    g_hat = F.logsigmoid(logits.float())
    return (-log_max * (1.0 - torch.exp(g_hat / log_max))).to(logits.dtype)

def _causal_conv1d_fallback(x: Tensor, weight: Tensor) -> Tensor:
    """Pure PyTorch depthwise causal conv1d. x: (B, H, L), weight: (H, k)."""
    k = weight.shape[1]
    x_padded = F.pad(x, (k - 1, 0))
    return F.conv1d(x_padded, weight.unsqueeze(1), groups=weight.shape[0])

try:
    # kernels>=0.16 requires callers to select a stable API version.  Omitting
    # it raises before loading and silently sends every training run through
    # the much slower PyTorch fallback below.
    kernel_module = get_kernel("kernels-community/causal-conv1d", version=1)
    _causal_conv1d_cuda_fn = kernel_module.causal_conv1d_fn

    def causal_conv1d_fn(x: Tensor, weight: Tensor) -> Tensor:
        # The Hub kernel registers CUDA only; keep CPU/reference execution
        # portable even when the extension imports successfully on a GPU host.
        if x.is_cuda:
            return _causal_conv1d_cuda_fn(x, weight)
        return _causal_conv1d_fallback(x, weight)
except Exception:
    print("causal-conv1d kernel not available, using PyTorch fallback")
    causal_conv1d_fn = _causal_conv1d_fallback


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('kernels-community/flash-attn3').flash_attn_interface
    except Exception:
        print("FA3 not available, falling back to PyTorch's scaled_dot_product_attention")
        return None

_fa3 = _load_fa3()

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training (FA3 only). q,k,v: (B, T, H, D)."""
    return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# --- Custom FP16 scaled matmul (avoids overflow) ---

@torch.library.custom_op("nanogpt::mm_fp16_scaled", mutates_args=())
def mm_fp16_scaled_op(
    x: Tensor, w: Tensor, x_s: Tensor, w_s: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Forward: y = (x / x_s) @ (w / w_s) * (x_s * w_s)
    Returns: y (FP32), x_scaled (FP16), w_scaled (FP16)
    x_s, w_s: scalar tensors (shape [])
    """
    @torch.compile
    def impl(x, w, x_s, w_s):
        assert x.is_contiguous()   # w can be non-contig (transposed)
        x_scaled = x.half() / x_s.half()
        w_scaled = w.half() / w_s.half()
        y_scaled = torch.matmul(x_scaled, w_scaled)
        y = y_scaled.float() * (x_s.float() * w_s.float())
        return y, x_scaled, w_scaled
    return impl(x, w, x_s, w_s)

@mm_fp16_scaled_op.register_fake
def _(x, w, x_s, w_s):
    return x @ w, x.half(), w.half()

# --- Backward op ---

@torch.library.custom_op("nanogpt::mm_fp16_scaled_backward", mutates_args=())
def mm_fp16_scaled_backward_op(
    grad_out: Tensor, x_scaled: Tensor, w_scaled: Tensor,
    x_s: Tensor, w_s: Tensor
) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad_out, x_scaled, w_scaled, x_s, w_s):
        grad_max = grad_out.abs().max()
        w_scaled_max = w_scaled.abs().max().float()
        K = w_scaled.shape[1] # inner dimension (out_features)

        # Safely bound the maximum value of the intermediate grad_x_scaled
        # so that `grad_scaled @ w_scaled.T` never overflows FP16's max value (~65500)
        factor = torch.clamp(w_scaled_max * K, min=1.0)
        grad_s = torch.clamp((grad_max / 65000.0) * factor, min=1e-6)

        # Divide in FP32 first to prevent FP16 underflow of grad_s, then cast
        grad_scaled = (grad_out / grad_s).half()

        # grad_x = (grad_scaled * grad_s) @ (w_scaled * w_s).T
        grad_x_scaled = torch.matmul(grad_scaled, w_scaled.T)
        grad_x = grad_x_scaled.float() * (grad_s.float() * w_s.float())

        # grad_w is accurately accumulated in FP32 (which is good practice)
        x_s_f = x_s.float()
        x_f = x_scaled.float() * x_s_f
        grad_w = torch.matmul(x_f.T, grad_out)  

        return grad_x.contiguous(), grad_w.contiguous()

    return impl(grad_out, x_scaled, w_scaled, x_s, w_s)
    
@mm_fp16_scaled_backward_op.register_fake
def _(g, x_scaled, w_scaled, *_):
    # grad_x is same shape & stride as x_scaled
    # grad_w has shape (in_features, out_features), must be row-major
    grad_x_fake = x_scaled.float().contiguous()
    grad_w_fake = w_scaled.float().contiguous()
    return grad_x_fake, grad_w_fake

# --- Autograd setup ---

def backward_t(ctx, grad_out, *_):
    x_scaled, w_scaled, x_s, w_s = ctx.saved_tensors
    grad_x, grad_w = torch.ops.nanogpt.mm_fp16_scaled_backward(
        grad_out, x_scaled, w_scaled, x_s, w_s
    )
    return grad_x, grad_w, None, None

def setup_context_t(ctx, inputs, output):
    x, w, x_s, w_s = inputs
    _, x_scaled, w_scaled = output
    ctx.save_for_backward(x_scaled, w_scaled, x_s, w_s)
    ctx.set_materialize_grads(False)

mm_fp16_scaled_op.register_autograd(backward_t, setup_context=setup_context_t)

# --- Custom FP8 matmul ---

@torch.library.custom_op("nanogpt::mm_fp8_t", mutates_args=())
def mm_fp8_t_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """Computes y = x @ w with F8 weights stored as (in_features, out_features)."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]  # x: (batch, in), w: (in, out)

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)

        # _scaled_mm requires column-major B. w_f8 is row-major (in, out).
        # .T.contiguous().T creates a column-major view without changing logical shape.
        w_f8_col_major = w_f8.T.contiguous().T

        out = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_fp8_t_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_fp8_t_backward", mutates_args=())
def mm_fp8_t_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()

        x_scale = grad.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)

        # grad_x = grad @ w.T
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        # grad_w = x.T @ grad
        # Result is (in, out), naturally matching weight storage. No final .T needed.
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, grad_w

    grad_x, grad_w = impl(g, x_f8, w_f8)

    return grad_x, grad_w

@mm_fp8_t_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward_t(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_fp8_t_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context_fp8_t(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_fp8_t_op.register_autograd(backward_t, setup_context=setup_context_fp8_t)


class LinearFP16(nn.Linear):
    """Use this for device without BF16 support, activation must be FP32"""
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        max_x = x_flat.detach().abs().max()
        max_w = self.weight.data.abs().max()

        # raw scales to keep inputs just inside FP16 range
        x_s_raw = torch.clamp(max_x / 65000.0, min=1e-8)
        w_s_raw = torch.clamp(max_w / 65000.0, min=1e-8)

        K = x_flat.shape[-1]
        # worst-case dot product magnitude before scaling up the scales
        worst_dot = (max_x / x_s_raw) * (max_w / w_s_raw) * K
        safety_factor = torch.sqrt(torch.clamp(worst_dot / 65504, min=1.0))
        x_s = x_s_raw * safety_factor
        w_s = w_s_raw * safety_factor

        y_flat, _, _ = torch.ops.nanogpt.mm_fp16_scaled(
            x_flat, self.weight.T, x_s, w_s
        )

        y = y_flat.view(*orig_shape[:-1], -1)
        if self.bias is not None:
            y = y + self.bias.type_as(y)
        return y


class LinearFP8(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features)
    """
    def __init__(self, in_features: int, out_features: int, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight)

    def forward(self, x: Tensor):
        _x = x.flatten(0, -2)
        out = torch.ops.nanogpt.mm_fp8_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
        out = out.reshape(*x.shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias.type_as(out)
        return out


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), None)


class LFM2Conv(AtmaConvBase):
    """Liquid Foundation Model 2 gated short convolutions
    (LFM2 Report, https://arxiv.org/abs/2511.23404)."""

    def __init__(self, dim: int, kernel_size=3):
        super().__init__(dim, linear_cls=Linear, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        projected = self.in_proj(x)  # (B, L, 3*H)
        B, C, x_proj = projected.chunk(3, dim=-1)

        x_gated = B * x_proj

        x_conv_input = x_gated.transpose(1, 2)  # (B, H, L)
        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2)).to(dtype=x_conv_input.dtype)
        x_conv = causal_conv1d_fn(x_conv_input.contiguous(), conv_weights)
        x_conv = x_conv.transpose(1, 2)  # (B, L, H)

        x_gated_2 = C * x_conv

        return self.out_proj(x_gated_2), torch.tensor(0.0, device=x.device)


class Rotary(nn.Module):
    """Half-truncate RoPE with base-frequency tuning (modded-nanogpt style). Applied to
    QK-normed q/k of shape (B, T, H, head_dim)."""

    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(AtmaAttnBase):
    """Softmax attention core for the ablation grid, three position schemes:
      pos="nope" -> canon convs on q/k/v, NO positional encoding (the legacy default).
      pos="rope" -> rotary on q/k, NO canon, tuned SDPA scale 0.12.
      pos="wall" -> canon convs + Wall Attention (Tilde Research): data-dependent
                    per-channel log-decay gates g, cumulative prefix sum P, score
                    q_i.k_j.exp(P_i-P_j) per channel, then softmax.
    Shares the GQA + output-gate surround with PolarAttention. The optional training
    sliding window (SWA), MSE distractor, and additive Titans memory branch are wired
    here so every grid cell (reg x distractor x memory x window x core) is distinct.
    With pos="nope", window=None, mem_enabled=False, num_random_keys in {0,None}, this is
    byte-identical to the original CausalSelfAttention."""

    def __init__(self, dim: int, head_dim=128, num_kv_heads: int = None, num_random_keys: int = None,
                 kernel_size=4, pos: str = "nope", window: int = None,
                 mem_enabled: bool = False, mem_chunk: int = 64,
                 mem_gamma_bias: float = 3.9, mem_beta_bias: float = 0.0, mem_kernel: str = "auto",
                 wall_gate_bias: float | None = None):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)
        self.num_random_keys = num_random_keys or 0
        self.pos = pos
        self.window = window
        self.sdpa_scale = 0.12 if pos == "rope" else None   # rope: tuned scale; nope/wall: SDPA default (1/sqrt(dk))
        self.rotary = Rotary(head_dim) if pos == "rope" else None
        if pos == "rope":
            # rope uses rotary, not canon -> drop the unused canon convs so they aren't
            # gradient-less params (Muon's lerp_ crashes on p.grad=None). Also makes rope a
            # true no-canon baseline.
            self.canon_q = self.canon_k = self.canon_v = None
        H, dk = self.num_heads, self.head_dim
        # Wall keeps canon (so all params are used); g is passed to the kernel directly.
        self.w_wall = LinearNoBias(dim, H * dk) if pos == "wall" else None
        if self.w_wall is not None:
            nn.init.zeros_(self.w_wall.weight)              # bias keeps init near vanilla softmax attention
        self.wall_gate_bias = _WALL_GATE_BIAS_INIT if wall_gate_bias is None else float(wall_gate_bias)
        self.mem = (TitansMemory(dim, H, dk, Linear, chunk=mem_chunk,
                                 gamma_bias=mem_gamma_bias, beta_bias=mem_beta_bias, kernel=mem_kernel)
                    if mem_enabled else None)

    def _sdpa(self, qh, kh, vh, W, scale):
        """Causal SDPA with an optional sliding-window band. q/k/v: (B, T, H, dk)."""
        T = qh.shape[1]
        q_t = qh.transpose(1, 2).contiguous()
        k_t = kh.transpose(1, 2).contiguous()
        v_t = vh.transpose(1, 2).contiguous()
        if W is None and q_t.is_cuda and sdpa_kernel is not None:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, scale=scale).transpose(1, 2)

        attn_mask, is_causal = None, True
        if W is not None:
            qi = torch.arange(T, device=qh.device).view(T, 1)
            ki = torch.arange(T, device=qh.device).view(1, T)
            band = (ki <= qi) & (ki > qi - W)
            attn_mask = torch.zeros(T, T, device=qh.device, dtype=qh.dtype).masked_fill(~band, float("-inf"))
            is_causal = False
        return F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attn_mask, is_causal=is_causal, scale=scale).transpose(1, 2)

    def _wall_attention(self, x, q_attn, k_attn, v_attn, groups, W):
        """Wall attention on canon'd q/k/v. q_attn:(B,T,H,dk); k/v_attn:(B,T,kvH,dk). Returns
        (y (B,T,H,dk), align_loss). Faithful path = Tilde's fused Triton kernel, including
        backward. The torch fallback is only for CPU / missing-kernel development runs."""
        B, T = x.shape[0], x.shape[1]
        H, dk = self.num_heads, self.head_dim
        g_logits = self.w_wall(x).view(B, T, H, dk) + self.wall_gate_bias
        g = _wall_log_decay(g_logits)                       # natural-log decay in (-g_max, 0]
        scale = dk ** -0.5
        align_loss = torch.tensor(0.0, device=x.device)

        if self.training and q_attn.is_cuda and not _HAS_WALL:
            raise RuntimeError("attn_type='wall' training on CUDA requires the wall_attn Triton package")

        if _HAS_WALL and q_attn.is_cuda:
            try:                                                    # fused fwd+bwd kernel; supports GQA
                wall_window = 0 if W is None else int(W)
                y = _wall_attn(q_attn.contiguous(), k_attn.contiguous(), v_attn.contiguous(),
                               g.to(q_attn.dtype).contiguous(), scale, wall_window)
                if self.num_random_keys > 0 and self.training:
                    R = self.num_random_keys
                    rand_input = torch.randn(B, R, x.shape[2], device=x.device, dtype=x.dtype)
                    k_rand = F.rms_norm(self.k(rand_input).view(B, R, self.num_kv_heads, dk), (dk,)).detach()
                    v_rand = self.v(rand_input).view(B, R, self.num_kv_heads, dk).detach()
                    q_pad = torch.zeros(B, R, H, dk, device=x.device, dtype=q_attn.dtype)
                    g_pad = torch.zeros(B, R, H, dk, device=x.device, dtype=g.dtype)
                    y_dist = _wall_attn(
                        torch.cat([q_pad, q_attn], dim=1).contiguous(),
                        torch.cat([k_rand, k_attn], dim=1).contiguous(),
                        torch.cat([v_rand, v_attn], dim=1).contiguous(),
                        torch.cat([g_pad, g], dim=1).to(q_attn.dtype).contiguous(),
                        scale,
                        0,
                    )[:, R:]
                    align_loss = F.mse_loss(y_dist, y)
                return y, align_loss
            except Exception as e:
                # Non-reentrant checkpoint uses this private exception for control flow during
                # recomputation. Do not wrap it as a Wall kernel failure.
                if e.__class__.__name__ == "_StopRecomputationError":
                    raise
                if self.training:
                    raise RuntimeError("wall_attn Triton kernel failed during training") from e
                pass                                                # eval can still fall through to the torch path

        k_exp = k_attn.repeat_interleave(groups, dim=2)             # (B,T,H,dk)
        v_exp = v_attn.repeat_interleave(groups, dim=2)
        g = g.float()
        P = torch.cumsum(g, dim=1)                                  # (B,T,H,dk), monotone decreasing
        P = P - 0.5 * (P.amax(1, keepdim=True) + P.amin(1, keepdim=True))   # recenter per (B,H,dk)
        P = P.clamp(-30.0, 30.0)                                    # NaN guard (only bites at long-ctx eval)
        q_scale = torch.exp(P).to(q_attn.dtype)
        k_scale = torch.exp(-P).to(k_exp.dtype)
        qt = q_attn * q_scale
        kt = k_exp * k_scale
        y = self._sdpa(qt, kt, v_exp, W, scale)
        return y, align_loss

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        H, dk = self.num_heads, self.head_dim
        groups = H // self.num_kv_heads

        q_gate = self.q(x).view(B, T, H, dk * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, dk)
        v = self.v(x).view(B, T, self.num_kv_heads, dk)

        # QK-Norm (per head)
        q = F.rms_norm(q, (dk,))
        k = F.rms_norm(k, (dk,))

        if self.pos == "rope":
            # rotary positions, no canon
            q_attn, k_attn, v_attn = self.rotary(q), self.rotary(k), v
            # memory is content-addressable -> feed it the pre-rotary (QK-normed) q/k/v
            q_mem, k_mem, v_mem = q, k, v
        else:
            # Canon (horizontal residual conv), no positional encoding
            q_conv_in = q.reshape(B, T, -1).transpose(1, 2)
            k_conv_in = k.reshape(B, T, -1).transpose(1, 2)
            v_conv_in = v.reshape(B, T, -1).transpose(1, 2)
            w_q = self.canon_q.weight.squeeze(1).to(dtype=q_conv_in.dtype)
            w_k = self.canon_k.weight.squeeze(1).to(dtype=k_conv_in.dtype)
            w_v = self.canon_v.weight.squeeze(1).to(dtype=v_conv_in.dtype)
            q_attn = (q_conv_in + causal_conv1d_fn(q_conv_in.contiguous(), w_q)).transpose(1, 2).reshape(B, T, H, dk)
            k_attn = (k_conv_in + causal_conv1d_fn(k_conv_in.contiguous(), w_k)).transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)
            v_attn = (v_conv_in + causal_conv1d_fn(v_conv_in.contiguous(), w_v)).transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)
            q_mem, k_mem, v_mem = q_attn, k_attn, v_attn

        W = self.window
        if self.pos == "wall":
            y, align_loss = self._wall_attention(x, q_attn, k_attn, v_attn, groups, W)
        else:
            align_loss = torch.tensor(0.0, device=x.device)
            if W is None and self.pos == "nope" and _fa3 is not None:
                # fast path: FA3 GQA causal (nope, no window) — preserves the legacy behavior
                y = flash_attn.flash_attn_func(q_attn, k_attn, v_attn, causal=True)
            else:
                k_sdpa = k_attn.repeat_interleave(groups, dim=2)
                v_sdpa = v_attn.repeat_interleave(groups, dim=2)
                attn_mask, is_causal = None, True
                if W is not None:
                    # sliding window band: query i attends to keys (i-W, i]  (training-only toggle)
                    qi = torch.arange(T, device=x.device).view(T, 1)
                    ki = torch.arange(T, device=x.device).view(1, T)
                    band = (ki <= qi) & (ki > qi - W)
                    attn_mask = torch.zeros(T, T, device=x.device, dtype=q_attn.dtype).masked_fill(~band, float("-inf"))
                    is_causal = False
                y = F.scaled_dot_product_attention(
                    q_attn.transpose(1, 2), k_sdpa.transpose(1, 2), v_sdpa.transpose(1, 2),
                    attn_mask=attn_mask, is_causal=is_causal, scale=self.sdpa_scale,
                ).transpose(1, 2)

            # Distractor (MSE): random keys must not perturb the output (noise rejection).
            if self.num_random_keys > 0 and self.training:  # skip during eval: custom mask forces O(T^2) memory
                R = self.num_random_keys
                rand_input = torch.randn(B, R, D, device=x.device, dtype=x.dtype)
                k_rand = self.k(rand_input).view(B, R, self.num_kv_heads, dk).detach()
                v_rand = self.v(rand_input).view(B, R, self.num_kv_heads, dk).detach()
                if self.pos == "rope":
                    k_rand = F.rms_norm(k_rand, (dk,))      # match real-key norm; noise carries no position
                k_dist = torch.cat([k_rand, k_attn], dim=1)
                v_dist = torch.cat([v_rand, v_attn], dim=1)
                # queries attend freely to all R distractors; causal only over the T real keys
                dist_mask = torch.zeros(T, R + T, device=x.device, dtype=q_attn.dtype)
                dist_mask[:, R:] = torch.triu(
                    torch.full((T, T), float("-inf"), device=x.device, dtype=q_attn.dtype), diagonal=1)
                k_sdpa = k_dist.repeat_interleave(groups, dim=2)
                v_sdpa = v_dist.repeat_interleave(groups, dim=2)
                y_dist = F.scaled_dot_product_attention(
                    q_attn.transpose(1, 2), k_sdpa.transpose(1, 2), v_sdpa.transpose(1, 2),
                    attn_mask=dist_mask, scale=self.sdpa_scale,
                ).transpose(1, 2)
                align_loss = F.mse_loss(y_dist, y)

        y = y.reshape(B, T, H * dk)
        y = y * torch.sigmoid(gate.reshape(B, T, -1))
        out = self.proj(y)

        if self.mem is not None:                                  # additive Titans memory branch (MAG)
            out = out + self.mem(x, q_mem.transpose(1, 2),
                                 k_mem.repeat_interleave(groups, dim=2).transpose(1, 2),
                                 v_mem.repeat_interleave(groups, dim=2).transpose(1, 2))
        return out, align_loss


class PolarAttention(AtmaAttnBase):
    """Training Polar attention: direction + bounded count channels.

    Same projections / Canon conv / GQA as CausalSelfAttention; the SDPA core is
    replaced by the validated polar_reduce (length-invariant direction + bounded
    count), and a per-head count channel is injected additively. The distractor
    objective calibrates the null sink (and Q geometry) to reject random keys.
    """

    def __init__(self, dim: int, head_dim=128, num_kv_heads: int = None, num_random_keys: int = None, kernel_size=4,
                 online: bool = False, k_block: int = 512, attn_kernel: str = "torch",
                 window: int = None, mem_enabled: bool = False, mem_chunk: int = 64,
                 mem_gamma_bias: float = 3.9, mem_beta_bias: float = 0.0, mem_kernel: str = "auto"):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, num_kv_heads=num_kv_heads, kernel_size=kernel_size)
        self.num_random_keys = num_random_keys or 0
        self.online = online            # stream keys in blocks -> O(T*k_block) memory (fwd+bwd)
        self.k_block = k_block
        self.window = window            # trainable sliding window (config); eval.py --window overrides
        self.attn_kernel = attn_kernel  # "torch" | "triton" (Triton flash kernel, CUDA only)
        self.polar_variant = "full"
        H, dk = self.num_heads, self.head_dim
        self.mu_proj = Linear(H, dim)                                   # count channel -> residual
        self.v_null = nn.Parameter(torch.zeros(H, dk))                 # default direction (null sink)
        self.null_base = nn.Parameter(torch.full((H,), _NULL_BASE_INIT))
        self.null_slope_raw = nn.Parameter(torch.full((H,), _NULL_SLOPE_INIT))
        self.len_gain_raw = nn.Parameter(torch.full((H,), _LEN_GAIN_INIT))
        self.mag_beta_raw = nn.Parameter(torch.full((H,), _MAG_BETA_INIT))
        # MAG long-term memory branch (additive 3rd channel). None unless enabled.
        self.mem = (TitansMemory(dim, H, dk, Linear, chunk=mem_chunk,
                                 gamma_bias=mem_gamma_bias, beta_bias=mem_beta_bias, kernel=mem_kernel)
                    if mem_enabled else None)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        H, dk = self.num_heads, self.head_dim

        q_gate = self.q(x).view(B, T, H, dk * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, dk)
        v = self.v(x).view(B, T, self.num_kv_heads, dk)

        q = F.rms_norm(q, (dk,))
        k = F.rms_norm(k, (dk,))

        q_conv_in = q.reshape(B, T, -1).transpose(1, 2)
        k_conv_in = k.reshape(B, T, -1).transpose(1, 2)
        v_conv_in = v.reshape(B, T, -1).transpose(1, 2)

        w_q = self.canon_q.weight.squeeze(1).to(dtype=q_conv_in.dtype)
        w_k = self.canon_k.weight.squeeze(1).to(dtype=k_conv_in.dtype)
        w_v = self.canon_v.weight.squeeze(1).to(dtype=v_conv_in.dtype)

        q_conv_out = q_conv_in + causal_conv1d_fn(q_conv_in.contiguous(), w_q)
        k_conv_out = k_conv_in + causal_conv1d_fn(k_conv_in.contiguous(), w_k)
        v_conv_out = v_conv_in + causal_conv1d_fn(v_conv_in.contiguous(), w_v)

        q_attn = q_conv_out.transpose(1, 2).reshape(B, T, H, dk)
        k_attn = k_conv_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)
        v_attn = v_conv_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)

        groups = H // self.num_kv_heads
        k_attn = k_attn.repeat_interleave(groups, dim=2)
        v_attn = v_attn.repeat_interleave(groups, dim=2)

        q_t = q_attn.transpose(1, 2)                       # (B, H, T, dk)
        k_t = k_attn.transpose(1, 2)
        v_t = v_attn.transpose(1, 2)

        n_keys = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
        polar_params = dict(
            v_null=self.v_null,
            null_base=self.null_base,
            null_slope_raw=(torch.full_like(self.null_slope_raw, -30.0)
                            if self.polar_variant == "fixed_null" else self.null_slope_raw),
            len_gain_raw=(torch.full_like(self.len_gain_raw, -30.0)
                          if self.polar_variant == "fixed_temperature" else self.len_gain_raw),
            mag_beta_raw=self.mag_beta_raw,
        )

        use_triton = (self.attn_kernel == "triton" and HAS_TRITON
                      and polar_attention_triton is not None and q_t.is_cuda)
        # window=None -> full causal; window=W -> trainable sliding band (each query sees
        # its last W keys). All three cores now model the band in the backward (Step 2).
        W = self.window
        if use_triton:
            # FlashAttention-style Triton kernel: O(T*block) memory, fused fwd+bwd.
            c, mag = polar_attention_triton(q_t, k_t, v_t, n_keys, window=W, **polar_params)
        elif self.online:
            # Memory-efficient: never materializes the (T, T) score matrix.
            c, mag = polar_attention_online(q_t, k_t, v_t, n_keys, k_block=self.k_block, window=W, **polar_params)
        else:
            sigma = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dk ** 0.5)
            sigma = sigma + torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=sigma.dtype), diagonal=1)
            n_temp = n_keys
            if W is not None:
                kidx = torch.arange(T, device=x.device)
                band = kidx.view(1, 1, 1, T) < (n_keys.view(1, 1, T, 1) - W)   # older than window
                sigma = sigma.masked_fill(band, float("-inf"))
                n_temp = torch.minimum(n_keys, n_keys.new_tensor(float(W)))
            c, mag = polar_reduce(sigma, v_t, n_temp, **polar_params)

        # Distractor (O(T*R), memory-friendly): random keys must lose to the null sink,
        # calibrating null_base/null_slope (+ Q geometry). Compares random keys vs the
        # null floor only — real keys staying above the floor is the task loss's job.
        align_loss = torch.tensor(0.0, device=x.device)
        if self.num_random_keys > 0 and self.training:
            R = self.num_random_keys
            rand_input = torch.randn(B, R, D, device=x.device, dtype=x.dtype)
            k_rand = F.rms_norm(self.k(rand_input).view(B, R, self.num_kv_heads, dk), (dk,)).detach()
            k_rand = k_rand.repeat_interleave(groups, dim=2).transpose(1, 2)   # (B, H, R, dk)
            sig_rand = torch.matmul(q_t, k_rand.transpose(-2, -1)) / (dk ** 0.5)  # (B, H, T, R)
            temp, null = polar_temp_null(n_keys, self.len_gain_raw, self.null_base, self.null_slope_raw)
            logits_r = (sig_rand * temp).float()
            null_col = (null * temp).expand(B, H, T, 1).float()
            w_r = torch.softmax(torch.cat([logits_r, null_col], dim=-1), dim=-1)
            align_loss = w_r[..., :R].sum(-1).mean()                           # mass random keys steal

        c_flat = c.transpose(1, 2).reshape(B, T, H * dk)
        content = self.proj(c_flat * torch.sigmoid(gate.reshape(B, T, -1)))
        if self.polar_variant == "direction_only":
            count = torch.zeros_like(content)
        else:
            if self.polar_variant == "constant_magnitude":
                mag = torch.ones_like(mag)
            count = self.mu_proj(mag.transpose(1, 2))      # (B, T, H) -> (B, T, D)
        out = content + count
        if self.mem is not None:                            # MAG long-term memory branch
            out = out + self.mem(x, q_t, k_t, v_t)
        return out, align_loss


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attention: bool = True,
        reg_mode: str = "baseline",
        sketch_dim: int = 64,
        head_dim: int = 128,
        num_kv_heads: int = None,
        num_random_keys: int = None,
        attn_kernel_size: int = 4,
        conv_kernel_size: int = 3,
        attn_online: bool = False,
        attn_k_block: int = 512,
        attn_kernel: str = "torch",
        attn_type: str = "polar",
        attn_window: int = None,
        mem_enabled: bool = False,
        mem_chunk: int = 64,
        mem_gamma_bias: float = 3.9,
        mem_beta_bias: float = 0.0,
        mem_kernel: str = "auto",
        wall_gate_bias: float | None = None,
        polar_variant: str = "full",
    ):
        super().__init__()
        if not attention:
            self.attn = LFM2Conv(dim, kernel_size=conv_kernel_size)
        elif attn_type == "polar":
            self.attn = PolarAttention(dim, head_dim=head_dim, num_kv_heads=num_kv_heads, num_random_keys=num_random_keys,
                                       kernel_size=attn_kernel_size, online=attn_online, k_block=attn_k_block,
                                       attn_kernel=attn_kernel, window=attn_window, mem_enabled=mem_enabled,
                                       mem_chunk=mem_chunk, mem_gamma_bias=mem_gamma_bias, mem_beta_bias=mem_beta_bias,
                                       mem_kernel=mem_kernel)
            allowed = {"full", "direction_only", "constant_magnitude", "fixed_null", "fixed_temperature"}
            if polar_variant not in allowed:
                raise ValueError(f"unknown polar_variant={polar_variant!r}; expected one of {sorted(allowed)}")
            self.attn.polar_variant = polar_variant
        else:  # "nope" | "rope" | "wall" — softmax core with shared GQA+gate surround
            self.attn = CausalSelfAttention(dim, head_dim=head_dim, num_kv_heads=num_kv_heads,
                                            num_random_keys=num_random_keys, kernel_size=attn_kernel_size,
                                            pos=attn_type, window=attn_window, mem_enabled=mem_enabled,
                                            mem_chunk=mem_chunk, mem_gamma_bias=mem_gamma_bias,
                                            mem_beta_bias=mem_beta_bias, mem_kernel=mem_kernel,
                                            wall_gate_bias=wall_gate_bias)
        self.mlp = MLP(dim, linear_cls=Linear)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.reg_mode = reg_mode
        self.sketch_dim = sketch_dim

    def forward(self, x: Tensor):
        x_attn, align_loss = self.attn(self.norm1(x))
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        reg_loss = sigreg(x, self.reg_mode, self.sketch_dim)
        return x, reg_loss, align_loss


class Model(nn.Module):
    def __init__(self, config: AtmaConfig, reg_mode: str = "baseline", sketch_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size).bfloat16()
        self.blocks = nn.ModuleList([
            Block(
                config.hidden_size,
                attention=(i % 4 == 2),
                reg_mode=reg_mode,
                sketch_dim=sketch_dim,
                head_dim=config.head_dim,
                num_kv_heads=config.num_key_value_heads,
                num_random_keys=config.num_random_keys,
                attn_kernel_size=config.attn_kernel_size,
                conv_kernel_size=config.conv_kernel_size,
                attn_online=config.attn_online,
                attn_k_block=config.attn_k_block,
                attn_kernel=config.attn_kernel,
                attn_type=config.attn_type,
                attn_window=config.attn_window,
                mem_enabled=config.mem_enabled,
                mem_chunk=config.mem_chunk,
                mem_gamma_bias=config.mem_gamma_bias,
                mem_beta_bias=config.mem_beta_bias,
                mem_kernel=config.mem_kernel,
                wall_gate_bias=config.wall_gate_bias,
                polar_variant=config.polar_variant,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.proj = Linear(config.hidden_size, config.vocab_size)
        self.norm = RMSNorm(config.hidden_size)
        self.num_attn_layers = sum(1 for block in self.blocks if isinstance(block.attn, AtmaAttnBase))

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.embed(inputs)
        total_reg_loss = 0.0
        total_align_loss = 0.0
        for block in self.blocks:
            x, reg_loss, align_loss = block(x)
            total_reg_loss += reg_loss
            total_align_loss += align_loss
        logits = self.proj(self.norm(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"), (total_reg_loss / len(self.blocks)), (total_align_loss / self.num_attn_layers)
