import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from types import SimpleNamespace

from train.reg import sigreg
from kernels import get_kernel
from model.config import AtmaConfig
from model.layers import RMSNorm, MLP
from model.blocks import AtmaConvBase, AtmaAttnBase

def _causal_conv1d_fallback(x: Tensor, weight: Tensor) -> Tensor:
    """Pure PyTorch depthwise causal conv1d. x: (B, H, L), weight: (H, k)."""
    k = weight.shape[1]
    x_padded = F.pad(x, (k - 1, 0))
    return F.conv1d(x_padded, weight.unsqueeze(1), groups=weight.shape[0])

try:
    kernel_module = get_kernel("kernels-community/causal-conv1d")
    causal_conv1d_fn = kernel_module.causal_conv1d_fn
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
        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        x_conv = causal_conv1d_fn(x_conv_input.contiguous(), conv_weights)
        x_conv = x_conv.transpose(1, 2)  # (B, L, H)

        x_gated_2 = C * x_conv

        return self.out_proj(x_gated_2)


class CausalSelfAttention(AtmaAttnBase):
    def __init__(self, dim: int, head_dim=128, kernel_size=4):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(1)

        q_gate = self.q(x).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim)

        # QK-Norm (Per head)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # Reshape to (B, hdim/kv_hdim, T) safely using transpose
        q_conv_in = q.reshape(B, T, -1).transpose(1, 2)  # (B, hdim, T)
        k_conv_in = k.reshape(B, T, -1).transpose(1, 2)  # (B, kv_hdim, T)
        v_conv_in = v.reshape(B, T, -1).transpose(1, 2)

        # Extract weights for causal_conv1d_fn -> (hdim/kv_hdim, kernel_size)
        w_q = self.canon_q.weight.squeeze(1)
        w_k = self.canon_k.weight.squeeze(1)
        w_v = self.canon_v.weight.squeeze(1)

        # Apply Canon layer (Horizontal residual: h' = h + conv1d(h))
        q_conv_out = q_conv_in + causal_conv1d_fn(q_conv_in.contiguous(), w_q)
        k_conv_out = k_conv_in + causal_conv1d_fn(k_conv_in.contiguous(), w_k)
        v_conv_out = v_conv_in + causal_conv1d_fn(v_conv_in.contiguous(), w_v)

        # Reshape to (B, T, num_heads/num_kv_heads, head_dim) for attention
        q_attn = q_conv_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        k_attn = k_conv_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, self.head_dim)
        v_attn = v_conv_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, self.head_dim)

        if _fa3 is None:
            # SDPA requires equal head counts — expand KV heads for GQA
            groups = self.num_heads // self.num_kv_heads
            k_sdpa = k_attn.repeat_interleave(groups, dim=2)
            v_sdpa = v_attn.repeat_interleave(groups, dim=2)
            y = F.scaled_dot_product_attention(q_attn.transpose(1, 2), k_sdpa.transpose(1, 2), v_sdpa.transpose(1, 2), is_causal=True).transpose(1, 2)
        else:
            # FA3 supports GQA natively
            y = flash_attn.flash_attn_func(
                q_attn, k_attn, v_attn,
                causal=True
            )

        # Post-process, apply gating
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.reshape(B, T, -1))

        return self.proj(y)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attention: bool = True,
        reg_mode: str = "baseline",
        sketch_dim: int = 64,
        head_dim: int = 128,
        attn_kernel_size: int = 4,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.attn = (
            CausalSelfAttention(dim, head_dim=head_dim, kernel_size=attn_kernel_size)
            if attention
            else LFM2Conv(dim, kernel_size=conv_kernel_size)
        )
        self.mlp = MLP(dim, linear_cls=Linear)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.reg_mode = reg_mode
        self.sketch_dim = sketch_dim

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        reg_loss = sigreg(x, self.reg_mode, self.sketch_dim)
        return x, reg_loss


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
                attn_kernel_size=config.attn_kernel_size,
                conv_kernel_size=config.conv_kernel_size,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.proj = Linear(config.hidden_size, config.vocab_size)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.embed(inputs)
        total_reg_loss = 0.0
        for block in self.blocks:
            x, reg_loss = block(x)
            total_reg_loss += reg_loss
        logits = self.proj(self.norm(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"), (total_reg_loss / len(self.blocks))
