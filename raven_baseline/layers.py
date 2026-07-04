from __future__ import annotations

import math
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.layers import RMSNorm

try:
    from fla.ops.gsa import chunk_gsa as _chunk_raven
    from fla.ops.gsa import fused_recurrent_gsa as _fused_recurrent_raven
    _HAS_FLA_GSA = True
except Exception:
    _chunk_raven = None
    _fused_recurrent_raven = None
    _HAS_FLA_GSA = False


_fla_chunk_raven = None
_fla_fused_recurrent_raven = None
if _HAS_FLA_GSA:
    def _raven_chunk_raw(q, k, v, s, g, scale: float):
        return _chunk_raven(q=q, k=k, v=v, s=s, g=g, scale=scale)[0]

    def _raven_fused_recurrent_raw(q, k, v, s, g, scale: float):
        return _fused_recurrent_raven(q=q, k=k, v=v, s=s, g=g, scale=scale)[0]

    _fla_chunk_raven = _raven_chunk_raw
    _fla_fused_recurrent_raven = _raven_fused_recurrent_raw

    if os.environ.get("FLA_CUSTOM_OP", "0") == "1":
        try:
            @torch.library.custom_op("atma::raven_chunk_fwd", mutates_args=())
            def _raven_chunk_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  s: torch.Tensor, g: torch.Tensor, scale: float) -> torch.Tensor:
                return _raven_chunk_raw(q, k, v, s, g, scale)

            @_raven_chunk_fwd.register_fake
            def _(q, k, v, s, g, scale: float):
                return torch.empty_like(v)

            @torch.library.custom_op("atma::raven_chunk_bwd", mutates_args=())
            def _raven_chunk_bwd(grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                                  v: torch.Tensor, s: torch.Tensor, g: torch.Tensor,
                                  scale: float) -> list[torch.Tensor]:
                with torch.enable_grad():
                    ins = [t.detach().requires_grad_(True) for t in (q, k, v, s, g)]
                    o = _raven_chunk_raw(*ins, scale)
                    return list(torch.autograd.grad(o, ins, grad_o))

            @_raven_chunk_bwd.register_fake
            def _(grad_o, q, k, v, s, g, scale: float):
                return [torch.empty_like(t) for t in (q, k, v, s, g)]

            def _chunk_setup(ctx, inputs, output):
                q, k, v, s, g, scale = inputs
                ctx.save_for_backward(q, k, v, s, g)
                ctx.scale = scale

            def _chunk_backward(ctx, grad_o):
                return tuple(_raven_chunk_bwd(grad_o, *ctx.saved_tensors, ctx.scale)) + (None,)

            _raven_chunk_fwd.register_autograd(_chunk_backward, setup_context=_chunk_setup)
            _fla_chunk_raven = _raven_chunk_fwd

            @torch.library.custom_op("atma::raven_recurrent_fwd", mutates_args=())
            def _raven_recurrent_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     s: torch.Tensor, g: torch.Tensor, scale: float) -> torch.Tensor:
                return _raven_fused_recurrent_raw(q, k, v, s, g, scale)

            @_raven_recurrent_fwd.register_fake
            def _(q, k, v, s, g, scale: float):
                return torch.empty_like(v)

            @torch.library.custom_op("atma::raven_recurrent_bwd", mutates_args=())
            def _raven_recurrent_bwd(grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                                     v: torch.Tensor, s: torch.Tensor, g: torch.Tensor,
                                     scale: float) -> list[torch.Tensor]:
                with torch.enable_grad():
                    ins = [t.detach().requires_grad_(True) for t in (q, k, v, s, g)]
                    o = _raven_fused_recurrent_raw(*ins, scale)
                    return list(torch.autograd.grad(o, ins, grad_o))

            @_raven_recurrent_bwd.register_fake
            def _(grad_o, q, k, v, s, g, scale: float):
                return [torch.empty_like(t) for t in (q, k, v, s, g)]

            def _recurrent_setup(ctx, inputs, output):
                q, k, v, s, g, scale = inputs
                ctx.save_for_backward(q, k, v, s, g)
                ctx.scale = scale

            def _recurrent_backward(ctx, grad_o):
                return tuple(_raven_recurrent_bwd(grad_o, *ctx.saved_tensors, ctx.scale)) + (None,)

            _raven_recurrent_fwd.register_autograd(_recurrent_backward, setup_context=_recurrent_setup)
            _fla_fused_recurrent_raven = _raven_recurrent_fwd
        except Exception:
            _fla_chunk_raven = _raven_chunk_raw
            _fla_fused_recurrent_raven = _raven_fused_recurrent_raw


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), None if self.bias is None else self.bias.type_as(x))


class RavenAttention(nn.Module):
    """Raven routed slot-memory mixer with a slow torch fallback.

    The fast path delegates to FLA's GSA kernels, matching the public Raven implementation.
    The fallback keeps imports and smoke tests usable on machines without FLA; it is not
    intended for full-scale training.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        num_slots: int = 256,
        topk: int = 32,
        feature_map: str = "swish",
        decay_type: str = "Mamba2",
        router_score: str = "sigmoid",
        router_type: str = "lin",
        add_gumbel_noise: bool = True,
        bias_rmm: bool = False,
        gate_logit_normalizer: int = 8,
        mem_enabled: bool = False,
        mem_chunk: int = 128,
        mem_gamma_bias: float = 3.9,
        mem_beta_bias: float = 0.0,
        mem_kernel: str = "auto",
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if topk > num_slots:
            raise ValueError("topk must be <= num_slots")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_slots = num_slots
        self.topk = topk
        self.feature_map = feature_map
        self.decay_type = decay_type
        self.router_score = router_score
        self.router_type = router_type
        self.add_gumbel_noise = add_gumbel_noise
        self.bias_rmm = bias_rmm
        self.gate_logit_normalizer = gate_logit_normalizer
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        if decay_type == "Mamba2":
            self.a_proj = Linear(hidden_size, num_heads, bias=False)
            A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = torch.clamp(dt, min=dt_init_floor)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            self.dt_bias._no_weight_decay = True
        elif decay_type == "GLA":
            self.f_proj = Linear(hidden_size, num_heads * num_slots, bias=False)
        else:
            raise ValueError(f"unsupported decay_type={decay_type}")

        if router_type == "lin":
            self.r_proj = Linear(hidden_size, num_heads * num_slots, bias=False)
        elif router_type == "mlp":
            self.r_proj = nn.Sequential(
                Linear(hidden_size, hidden_size, bias=True),
                nn.GELU(),
                Linear(hidden_size, num_heads * num_slots, bias=False),
            )
        else:
            raise ValueError(f"unsupported router_type={router_type}")
        if bias_rmm:
            self.r_bias = nn.Parameter(torch.empty(num_heads, num_slots, dtype=torch.float32))
            nn.init.zeros_(self.r_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.o_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.o_proj = Linear(hidden_size, hidden_size, bias=False)

        if mem_enabled:
            from model.blocks import TitansMemory
            self.mem = TitansMemory(
                hidden_size, num_heads, self.head_dim, Linear, chunk=mem_chunk,
                gamma_bias=mem_gamma_bias, beta_bias=mem_beta_bias, kernel=mem_kernel,
            )
        else:
            self.mem = None

    def _feature_map(self, x: Tensor) -> Tensor:
        if self.feature_map == "swish":
            return x * torch.sigmoid(x)
        if self.feature_map == "relu":
            return F.relu(x)
        raise ValueError(f"unsupported feature_map={self.feature_map}")

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        router = self.r_proj(x).view(B, T, self.num_heads, self.num_slots)
        if self.add_gumbel_noise and self.training:
            router = router - torch.empty_like(router).exponential_().log()

        if self.router_score == "sigmoid":
            scores = torch.sigmoid(router)
        elif self.router_score == "softmax":
            scores = torch.softmax(router, dim=-1)
        else:
            raise ValueError(f"unsupported router_score={self.router_score}")
        rank_scores = scores + self.r_bias.float() if self.bias_rmm else scores
        route_idx = rank_scores.topk(self.topk, dim=-1).indices
        topk_weights = torch.gather(scores, dim=-1, index=route_idx)
        if self.router_score == "sigmoid":
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))

        if self.decay_type == "Mamba2":
            f = (-self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)).unsqueeze(-1)
        else:
            f = self.f_proj(x).view(B, T, self.num_heads, self.num_slots)
            f = F.logsigmoid(f) / self.gate_logit_normalizer
        f = (f * multihot).to(x.dtype)
        s = (1.0 - f.exp()).to(x.dtype)
        return f, s

    def _torch_raven(self, q: Tensor, k: Tensor, v: Tensor, f: Tensor, s: Tensor) -> Tensor:
        B, T, H, D = q.shape
        M = f.shape[-1]
        state_k = q.new_zeros(B, H, M, D)
        state_v = q.new_zeros(B, H, M, D)
        outs = []
        for t in range(T):
            decay = f[:, t].exp().transpose(1, 2).unsqueeze(-1)  # B,M,H,1
            write = s[:, t].transpose(1, 2).unsqueeze(-1)
            kt = k[:, t].unsqueeze(1)                            # B,1,H,D
            vt = v[:, t].unsqueeze(1)
            sk = state_k.transpose(1, 2)
            sv = state_v.transpose(1, 2)
            sk = decay * sk + write * kt
            sv = decay * sv + write * vt
            state_k = sk.transpose(1, 2)
            state_v = sv.transpose(1, 2)
            scores = torch.einsum("bhmd,bhd->bhm", state_k, q[:, t]) * self.scale
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            outs.append(torch.einsum("bhm,bhmd->bhd", weights, state_v))
        return torch.stack(outs, dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self._feature_map(q))
        k = self.k_norm(self._feature_map(k))
        v = F.silu(v)
        f, s = self._route(x)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        if _HAS_FLA_GSA and q.is_cuda:
            mode = "fused_recurrent" if T <= 64 else "chunk"
            if mode == "fused_recurrent":
                o = _fla_fused_recurrent_raven(q, k, v, s, f, self.scale)
            else:
                o = _fla_chunk_raven(q, k, v, s, f, self.scale)
        else:
            o = self._torch_raven(q, k, v, f, s)

        o = self.o_norm(F.silu(o)).reshape(B, T, self.hidden_size)
        out = self.o_proj(o)
        if self.mem is not None:
            out = out + self.mem(x, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return out, torch.tensor(0.0, device=x.device)
