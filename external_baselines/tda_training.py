"""Training-only optimized execution for the pinned TDA kernel.

The upstream Threshold Differential Attention wrapper reads its fixed threshold
coefficient with ``Tensor.item()`` once for each of its two attention paths.  In the
four-global-block training model that creates eight device/host synchronizations per
microbatch and prevents CUDA-graph capture.  This module keeps the pinned kernels and
their autograd implementation unchanged, but passes the configured threshold as a
Python scalar, compiles the tensor-only regions around the kernel, and captures the
fixed-shape forward/backward microstep in a CUDA graph.

The original model remains the sole checkpoint owner.  Helper modules below share its
Parameter objects but are deliberately kept outside its module tree.
"""
from __future__ import annotations

import gc

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import triton


class _TDATunedThreshold(torch.autograd.Function):
    """Pinned FP32 kernel arithmetic with launch geometry tuned for head_dim=64.

    This preserves the upstream wrapper's fp32 Q/K/V promotion, fp32 saved tensors,
    and output/gradient casts. Only the Triton tile sizes and pipeline stages change.
    """

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, beta: float, relu_power: float) -> Tensor:
        from triton_threshold_attention import _threshold_rela_fwd_kernel

        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("TDA Q, K, and V shapes must match")
        original_dtype = q.dtype
        if q.dtype != torch.float32:
            q, k, v = q.float(), k.float(), v.float()
        batch, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)
        block_m, block_n = 64, 32
        grid = (batch * num_heads, triton.cdiv(seq_len, block_m))
        _threshold_rela_fwd_kernel[grid](
            q, k, v, out, float(beta), seq_len, float(relu_power),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=head_dim,
            num_warps=4, num_stages=2,
        )
        if original_dtype != torch.float32:
            out = out.to(original_dtype)
        ctx.save_for_backward(q, k, v)
        ctx.beta = float(beta)
        ctx.relu_power = float(relu_power)
        ctx.head_dim = head_dim
        ctx.seq_len = seq_len
        ctx.original_dtype = original_dtype
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        from triton_threshold_attention import (
            _threshold_rela_bwd_kernel_dkv,
            _threshold_rela_bwd_kernel_dq,
        )

        q, k, v = ctx.saved_tensors
        batch, num_heads = q.shape[:2]
        if dout.dtype != torch.float32:
            dout = dout.float()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        block_m, block_n = 64, 16
        grid_dq = (batch * num_heads, triton.cdiv(ctx.seq_len, block_m))
        _threshold_rela_bwd_kernel_dq[grid_dq](
            q, k, v, dout, dq, ctx.beta, ctx.seq_len, ctx.relu_power,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            HEAD_DIM=ctx.head_dim, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=ctx.head_dim,
            num_warps=4, num_stages=2,
        )
        block_m = block_n = 64
        grid_dkv = (batch * num_heads, triton.cdiv(ctx.seq_len, block_n))
        _threshold_rela_bwd_kernel_dkv[grid_dkv](
            q, k, v, dout, dk, dv, ctx.beta, ctx.seq_len, ctx.relu_power,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            HEAD_DIM=ctx.head_dim, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=ctx.head_dim,
            num_warps=4, num_stages=1,
        )
        if ctx.original_dtype != torch.float32:
            dq = dq.to(ctx.original_dtype)
            dk = dk.to(ctx.original_dtype)
            dv = dv.to(ctx.original_dtype)
        return dq, dk, dv, None, None


def tuned_threshold_rela(q: Tensor, k: Tensor, v: Tensor, beta: float, relu_power: float) -> Tensor:
    """Numerically matched, launch-tuned adapter for the pinned Threshold ReLA kernels."""
    return _TDATunedThreshold.apply(q, k, v, beta, relu_power)


class _LocalBlock(nn.Module):
    """Complete LFM2 block without the disabled auxiliary-loss bookkeeping."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.input_norm = block.norm1
        self.mixer = block.attn
        self.mlp_norm = block.norm2
        self.mlp = block.mlp

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.mixer(self.input_norm(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        return x + self.mlp(self.mlp_norm(x))


class _TDAPrepare(nn.Module):
    """Outer norm, projections, layout conversion, and cosine normalization."""

    def __init__(self, block: nn.Module):
        super().__init__()
        mixer = block.attn
        self.input_norm = block.norm1
        self.q_proj = mixer.q_proj
        self.k_proj = mixer.k_proj
        self.v_proj = mixer.v_proj
        self.num_heads = mixer.num_heads
        self.head_dim = mixer.head_dim

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z = self.input_norm(x)
        batch, length, _ = z.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q1, q2 = self.q_proj(z).chunk(2, dim=-1)
        k1, k2 = self.k_proj(z).chunk(2, dim=-1)
        q1 = q1.view(shape).transpose(1, 2).contiguous()
        q2 = q2.view(shape).transpose(1, 2).contiguous()
        k1 = k1.view(shape).transpose(1, 2).contiguous()
        k2 = k2.view(shape).transpose(1, 2).contiguous()
        v = self.v_proj(z).view(shape).transpose(1, 2).contiguous()
        # TDA consumes cosine-normalized views, whereas the matched Titans side
        # channel consumes the raw views and performs its own in-kernel QK norm.
        return (
            q1, q2, k1, k2, v,
            F.normalize(q1, p=2, dim=-1),
            F.normalize(q2, p=2, dim=-1),
            F.normalize(k1, p=2, dim=-1),
            F.normalize(k2, p=2, dim=-1),
        )


class _TDAOutput(nn.Module):
    """Signed reduction normalization and output projection."""

    def __init__(self, block: nn.Module):
        super().__init__()
        mixer = block.attn
        self.lambda_param = mixer.lambda_param
        self.o_proj = mixer.o_proj
        self.head_dim = mixer.head_dim
        self.hidden_size = mixer.hidden_size

    def forward(self, out1: Tensor, out2: Tensor) -> Tensor:
        y = out1 - self.lambda_param.clamp(0.0, 1.0) * out2
        y = F.rms_norm(y, (self.head_dim,))
        batch, _, length, _ = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, self.hidden_size))


class _MemoryPrepare(nn.Module):
    """Projection/gate work surrounding the opaque FLA Titans recurrence."""

    def __init__(self, memory: nn.Module):
        super().__init__()
        self.w_gamma = memory.w_gamma
        self.w_beta = memory.w_beta
        self.gamma_bias = memory.gamma_bias
        self.beta_bias = memory.beta_bias

    def forward(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, ...]:
        g = F.logsigmoid(self.w_gamma(x).float() + self.gamma_bias)
        beta = torch.sigmoid(self.w_beta(x).float() + self.beta_bias)
        return (
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            g.contiguous(),
            beta.contiguous(),
        )


class _MemoryOutput(nn.Module):
    def __init__(self, memory: nn.Module):
        super().__init__()
        self.gate = memory.gate
        self.proj = memory.proj
        self.head_dim = memory.dk

    def forward(self, x: Tensor, recurrence: Tensor) -> Tensor:
        r = F.rms_norm(recurrence, (self.head_dim,))
        r = r.reshape(*r.shape[:2], -1).to(x.dtype)
        return self.proj(r * torch.sigmoid(self.gate(x)))


class _TDABlockTail(nn.Module):
    """Residual additions and complete MLP branch after TDA/Titans kernels."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.mlp_norm = block.norm2
        self.mlp = block.mlp

    def forward(self, residual: Tensor, attention: Tensor, memory: Tensor) -> Tensor:
        x = residual + attention + memory
        return x + self.mlp(self.mlp_norm(x))


class _LossHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.norm = model.norm
        self.proj = model.proj

    def forward(self, x: Tensor, targets: Tensor) -> Tensor:
        logits = self.proj(self.norm(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


class TDACUDAGraphTrainer:
    """Split-compile and capture a fixed-shape TDA forward/backward microstep."""

    def __init__(self, model: nn.Module):
        if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
            raise RuntimeError("TDA CUDA-graph training requires a CUDA model")
        if model.cfg.get("arch_type") != "tda_hybrid":
            raise ValueError("TDACUDAGraphTrainer requires arch_type='tda_hybrid'")
        if model.cfg.get("reg_mode") != "baseline":
            raise ValueError("optimized TDA training currently requires reg_mode='baseline'")
        if model.cfg.get("sigr_alpha", 0.0) != 0.0 or model.cfg.get("dist_align_loss_weight", 0.0) != 0.0:
            raise ValueError("optimized TDA training requires zero auxiliary-loss weights")

        self.model = model
        self.beta = float(model.cfg.get("tda_beta", 1.0))
        self.relu_power = float(model.cfg.get("tda_relu_power", 2.0))
        self.tuned_kernel = bool(model.cfg.get("tda_tuned_kernel", False))
        self.local_blocks: dict[int, nn.Module] = {}
        self.prepares: dict[int, nn.Module] = {}
        self.outputs: dict[int, nn.Module] = {}
        self.tails: dict[int, nn.Module] = {}
        self.memory_prepares: dict[int, nn.Module] = {}
        self.memory_outputs: dict[int, nn.Module] = {}
        self.memory_kernel = None
        self.kernels = {}
        for index, block in enumerate(model.blocks):
            if hasattr(block.attn, "q_proj") and hasattr(block.attn, "lambda_param"):
                self.prepares[index] = torch.compile(_TDAPrepare(block), fullgraph=True)
                self.outputs[index] = torch.compile(_TDAOutput(block), fullgraph=True)
                self.tails[index] = torch.compile(_TDABlockTail(block), fullgraph=True)
                self.kernels[index] = block.attn._kernel()
                if block.attn.mem is not None:
                    self.memory_prepares[index] = torch.compile(_MemoryPrepare(block.attn.mem), fullgraph=True)
                    self.memory_outputs[index] = torch.compile(_MemoryOutput(block.attn.mem), fullgraph=True)
            else:
                self.local_blocks[index] = torch.compile(_LocalBlock(block), fullgraph=True)
        if len(self.prepares) != 4:
            raise ValueError(f"optimized TDA runner expected four global blocks, found {len(self.prepares)}")
        if self.memory_prepares:
            from model.blocks import _FLA_IMPORT_ERROR, _HAS_FLA, _fla_gated_delta

            if not _HAS_FLA or _fla_gated_delta is None:
                raise RuntimeError(
                    "optimized TDA requires the pinned fused FLA Titans kernel; "
                    f"import failed with {_FLA_IMPORT_ERROR}"
                )
            self.memory_kernel = _fla_gated_delta

        self.loss_head = torch.compile(_LossHead(model), fullgraph=True)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_inputs: Tensor | None = None
        self.static_targets: Tensor | None = None
        self.captured_loss: Tensor | None = None

    def forward_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Execute the numerically matched split path without graph replay."""
        x = self.model.embed(inputs)
        for index, block in enumerate(self.model.blocks):
            if index in self.local_blocks:
                x = self.local_blocks[index](x)
                continue

            q1, q2, k1, k2, v, tq1, tq2, tk1, tk2 = self.prepares[index](x)
            # Calling each pinned path directly with a host scalar avoids the
            # upstream beta.item() synchronization while preserving its kernels.
            kernel_module = self.kernels[index].__module__
            module = __import__(kernel_module, fromlist=["threshold_rela_triton"])
            threshold = tuned_threshold_rela if self.tuned_kernel else module.threshold_rela_triton
            out1 = threshold(tq1, tk1, v, self.beta, self.relu_power)
            out2 = threshold(tq2, tk2, v, self.beta, self.relu_power)
            attention = self.outputs[index](out1, out2)
            if index in self.memory_prepares:
                mq, mk, mv, mg, mbeta = self.memory_prepares[index](x, q1, k1, v)
                assert self.memory_kernel is not None
                recurrence = self.memory_kernel(mq, mk, mv, mg, mbeta)
                memory = self.memory_outputs[index](x, recurrence)
            else:
                memory = torch.zeros_like(attention)
            x = self.tails[index](x, attention, memory)
        return self.loss_head(x, targets)

    def _capture(self, inputs: Tensor, targets: Tensor) -> None:
        self.static_inputs = torch.empty_like(inputs)
        self.static_targets = torch.empty_like(targets)
        self.static_inputs.copy_(inputs)
        self.static_targets.copy_(targets)

        warmup_loss = self.forward_loss(self.static_inputs, self.static_targets)
        warmup_loss.backward()
        torch.cuda.synchronize()
        self.model.zero_grad(set_to_none=False)
        del warmup_loss
        gc.collect()

        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
        if override is not None:
            override(True)
        try:
            with torch.cuda.graph(self.graph):
                self.captured_loss = self.forward_loss(self.static_inputs, self.static_targets)
                self.captured_loss.backward()
        finally:
            if override is not None:
                override(False)
        torch.cuda.synchronize()
        self.model.zero_grad(set_to_none=False)

    def backward(self, inputs: Tensor, targets: Tensor) -> None:
        """Accumulate gradients for one microbatch through CUDA-graph replay."""
        if self.graph is None:
            torch.cuda.reset_peak_memory_stats()
            self._capture(inputs, targets)
        assert self.static_inputs is not None and self.static_targets is not None
        if inputs.shape != self.static_inputs.shape or targets.shape != self.static_targets.shape:
            raise ValueError(
                "TDA CUDA graph has fixed input shapes: "
                f"expected {tuple(self.static_inputs.shape)}/{tuple(self.static_targets.shape)}, "
                f"got {tuple(inputs.shape)}/{tuple(targets.shape)}"
            )
        self.static_inputs.copy_(inputs)
        self.static_targets.copy_(targets)
        self.graph.replay()

    @property
    def captured(self) -> bool:
        return self.graph is not None
