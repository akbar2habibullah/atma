import torch
from torch import Tensor, nn
import torch.nn.functional as F

import reg

from kernels import get_kernel

kernel_module = get_kernel("kernels-community/causal-conv1d")
causal_conv1d_fn = kernel_module.causal_conv1d_fn

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

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

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

class LFM2Conv(nn.Module):
    """Liquid Foundation Model 2 gated short convolutions 
    (LFM2 Report, https://arxiv.org/abs/2511.23404)."""
    def __init__(self, dim: int, kernel_size=3):
        super().__init__()
        self.hidden_size = dim
        self.kernel_size = kernel_size

        # Input projection to gates and values
        self.in_proj = Linear(
            self.hidden_size,
            3 * self.hidden_size,  # B, C, x
        )

        # Short convolution
        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=self.hidden_size,  # Depthwise convolution for efficiency
            bias=False,
        )

        # Output projection
        self.out_proj = Linear(
            self.hidden_size, self.hidden_size,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_size = x.shape

        # Input projection: B, C, x = linear(x)
        projected = self.in_proj(x)  # (B, L, 3*H)
        B, C, x_proj = projected.chunk(3, dim=-1)  # Each: (B, L, H)

        # First gating: x = B*x
        x_gated = B * x_proj

        # Apply short convolution
        # Convert to (B, H, L) for conv1d
        x_conv_input = x_gated.transpose(1, 2)  # (B, H, L)
        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        x_conv = causal_conv1d_fn(x_conv_input.contiguous(), conv_weights)
        x_conv = x_conv.transpose(1, 2)  # (B, L, H)

        # Second gating: x = C*x
        x_gated_2 = C*x_conv

        # Output projection
        output = self.out_proj(x_gated_2)

        return output

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128, kernel_size=4):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        
        self.q = Linear(dim, hdim * 2) # projection shared with gate
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        
        # Canon-B applied to Q, K, and V (m=3d)
        self.canon_q = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)
        self.canon_k = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)
        self.canon_v = nn.Conv1d(hdim, hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=hdim, bias=False)
        
        self.proj = Linear(hdim, dim)

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(1)
        
        q_gate = self.q(x).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        
        # QK-Norm (Per head)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # Reshape to (B, hdim, T) safely using transpose
        q_conv_in = q.reshape(B, T, -1).transpose(1, 2)
        k_conv_in = k.reshape(B, T, -1).transpose(1, 2)
        v_conv_in = v.reshape(B, T, -1).transpose(1, 2)

        # Extract weights for causal_conv1d_fn -> (hdim, kernel_size)
        w_q = self.canon_q.weight.squeeze(1)
        w_k = self.canon_k.weight.squeeze(1)
        w_v = self.canon_v.weight.squeeze(1)

        # Apply Canon layer (Horizontal residual: h' = h + conv1d(h))
        q_conv_out = q_conv_in + causal_conv1d_fn(q_conv_in.contiguous(), w_q)
        k_conv_out = k_conv_in + causal_conv1d_fn(k_conv_in.contiguous(), w_k)
        v_conv_out = v_conv_in + causal_conv1d_fn(v_conv_in.contiguous(), w_v)

        # Reshape to (B, T, num_heads, head_dim) for FA3
        q_attn = q_conv_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        k_attn = k_conv_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        v_attn = v_conv_out.transpose(1, 2).reshape(B, T, self.num_heads, self.head_dim)
        
        if _fa3 is None:
            y = F.scaled_dot_product_attention(q_attn.transpose(1, 2), k_attn.transpose(1, 2), v_attn.transpose(1, 2), is_causal=True).transpose(1, 2)
        else:
            y = flash_attn.flash_attn_func(
                q_attn, k_attn, v_attn,
                causal=True
            )
        
        # Post-process, apply gating
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.reshape(B, T, -1))
        
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim * 2) # shared with gate
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x_gate = self.fc(x)
        x, gate = torch.chunk(x_gate, 2, dim=-1)
        x = gate * x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, attention: bool = True, reg_mode: str = "baseline", sigr_alpha: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(dim) if attention else LFM2Conv(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.reg_mode = reg_mode
        self.sigr_alpha = sigr_alpha

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        reg_loss = reg.sigreg(x, self.reg_mode, self.sigr_alpha)

        return x, reg_loss

class Model(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, reg_mode: str = "baseline", sigr_alpha: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16() # float() if using Linear16
        self.blocks = nn.ModuleList([Block(model_dim, attention=True) if i%4 == 2 else Block(model_dim, attention=False) for i in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.embed(inputs)
        total_reg_loss = 0.0
        for block in self.blocks:
            x, reg_loss = block(x)
            total_reg_loss += reg_loss
        logits = self.proj(self.norm(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"), (total_reg_loss / len(self.blocks))
