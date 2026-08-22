"""Training-only optimized execution for the pinned FLA GDN-2 layer.

FLA deliberately marks its short convolution, GDN-2 recurrence, and gated
normalization as compiler-disabled.  Compiling ``ExternalLM`` directly then
hits a graph break inside the block loop and leaves almost the entire model in
eager mode.  This module makes those three boundaries explicit, compiles the
pure-PyTorch work around them, and captures the fixed-shape forward/backward
microstep in a CUDA graph.

The original model remains the owner of every parameter.  These helper modules
only hold references to it, so checkpoint keys and evaluation execution are
unchanged.
"""
from __future__ import annotations

import gc

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _GDN2Prepare(nn.Module):
    """Outer RMSNorm plus projection/gate preparation before fused kernels."""

    def __init__(self, block: nn.Module):
        super().__init__()
        mixer = block.attn
        self.input_norm = block.norm1
        self.q_proj = mixer.q_proj
        self.k_proj = mixer.k_proj
        self.v_proj = mixer.v_proj
        self.f_proj = mixer.f_proj
        self.b_proj = mixer.b_proj
        self.w_proj = mixer.w_proj
        self.g_proj = mixer.g_proj
        self.A_log = mixer.A_log
        self.dt_bias = mixer.dt_bias
        self.num_heads = mixer.num_heads
        self.num_v_heads = mixer.num_v_heads
        self.head_k_dim = mixer.head_k_dim
        self.head_v_dim = mixer.head_v_dim
        self.allow_neg_eigval = mixer.allow_neg_eigval

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z = self.input_norm(x)
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        g = F.softplus(self.f_proj(z).float() + self.dt_bias)
        g = g.view(*g.shape[:-1], self.num_heads, self.head_k_dim)
        g = -self.A_log.float().exp()[None, None, :, None] * g
        b = self.b_proj(z).sigmoid().view(*z.shape[:-1], self.num_heads, self.head_k_dim)
        w = self.w_proj(z).sigmoid().view(*z.shape[:-1], self.num_v_heads, self.head_v_dim)
        if self.allow_neg_eigval:
            b = b * 2.0
        gate = self.g_proj(z).view(*z.shape[:-1], self.num_v_heads, self.head_v_dim)
        return q, k, v, g, b, w, gate


class _GDN2BlockTail(nn.Module):
    """Output projection, residual connection, and the complete MLP branch."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.o_proj = block.attn.o_proj
        self.mlp_norm = block.norm2
        self.mlp = block.mlp

    def forward(self, residual: Tensor, mixed: Tensor) -> Tensor:
        x = residual + self.o_proj(mixed.flatten(-2))
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


class GDN2CUDAGraphTrainer:
    """Capture and replay a fixed-shape GDN-2 forward/backward microstep."""

    def __init__(self, model: nn.Module):
        if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
            raise RuntimeError("GDN-2 CUDA-graph training requires a CUDA model")
        if model.cfg.get("arch_type") != "gdn2_native":
            raise ValueError("GDN2CUDAGraphTrainer requires arch_type='gdn2_native'")
        if model.cfg.get("reg_mode") != "baseline":
            raise ValueError("optimized GDN-2 training currently requires reg_mode='baseline'")
        if model.cfg.get("sigr_alpha", 0.0) != 0.0 or model.cfg.get("dist_align_loss_weight", 0.0) != 0.0:
            raise ValueError("optimized GDN-2 training requires zero auxiliary-loss weights")

        for block in model.blocks:
            mixer = block.attn
            if not mixer.use_short_conv:
                raise ValueError("optimized GDN-2 training expects the configured short convolutions")
            if mixer.num_heads != mixer.num_v_heads:
                raise ValueError("optimized GDN-2 training does not yet support grouped value heads")

        self.model = model
        # Keep these helpers outside the model module tree. They share the
        # original Parameter objects but must never alter checkpoint names.
        self.prepares = [torch.compile(_GDN2Prepare(block), fullgraph=True) for block in model.blocks]
        self.tails = [torch.compile(_GDN2BlockTail(block), fullgraph=True) for block in model.blocks]
        self.loss_head = torch.compile(_LossHead(model), fullgraph=True)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_inputs: Tensor | None = None
        self.static_targets: Tensor | None = None
        self.captured_loss: Tensor | None = None

    def forward_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Execute the split-compiled forward path without graph capture."""
        from fla.ops.gdn2 import chunk_gdn2

        x = self.model.embed(inputs)
        for block, prepare, tail in zip(self.model.blocks, self.prepares, self.tails):
            mixer = block.attn
            q, k, v, g, b, w, gate = prepare(x)
            q = mixer.q_conv1d(x=q, cache=None, output_final_state=False, cu_seqlens=None)[0]
            k = mixer.k_conv1d(x=k, cache=None, output_final_state=False, cu_seqlens=None)[0]
            v = mixer.v_conv1d(x=v, cache=None, output_final_state=False, cu_seqlens=None)[0]
            q = q.view(*q.shape[:-1], mixer.num_heads, mixer.head_k_dim)
            k = k.view(*k.shape[:-1], mixer.num_heads, mixer.head_k_dim)
            v = v.view(*v.shape[:-1], mixer.num_v_heads, mixer.head_v_dim)
            mixed, _ = chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                use_qk_l2norm_in_kernel=True,
            )
            mixed = mixer.o_norm(mixed, gate)
            x = tail(x, mixed)
        return self.loss_head(x, targets)

    def _capture(self, inputs: Tensor, targets: Tensor) -> None:
        self.static_inputs = torch.empty_like(inputs)
        self.static_targets = torch.empty_like(targets)
        self.static_inputs.copy_(inputs)
        self.static_targets.copy_(targets)

        # Compile every helper and allocate persistent gradient buffers before
        # capture. Keeping gradients allocated is required for subsequent
        # AccumulateGrad nodes to replay into the same addresses.
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
                "GDN-2 CUDA graph has fixed input shapes: "
                f"expected {tuple(self.static_inputs.shape)}/{tuple(self.static_targets.shape)}, "
                f"got {tuple(inputs.shape)}/{tuple(targets.shape)}"
            )
        self.static_inputs.copy_(inputs)
        self.static_targets.copy_(targets)
        self.graph.replay()

    @property
    def captured(self) -> bool:
        return self.graph is not None
