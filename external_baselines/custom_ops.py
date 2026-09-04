"""Compile-opaque wrappers for the pinned Mamba-3 training kernels.

Mamba exposes its Triton kernels through ``torch.autograd.Function``. Those functions
cannot be compiled by the torch/triton versions pinned for this experiment. The wrappers
below follow the established ``model.blocks`` pattern: fused forward and backward calls
are opaque custom operators, while projections, residuals, and MLPs stay in the graph.

Only the stateless SISO full-sequence path used for training is wrapped. Cache/state and
variable-length paths continue to call the pinned upstream implementation.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import torch


_MAMBA3_OP: Callable | None = None
_MAMBA3_NORM_OP: Callable | None = None
_PATCHED: set[str] = set()


def _contiguous_gradients(inputs: list[torch.Tensor], gradients) -> list[torch.Tensor]:
    """Return a stable real/fake layout contract for Inductor custom-op outputs."""
    results = []
    for tensor, value in zip(inputs, gradients):
        result = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if value is None:
            result.zero_()
        else:
            result.copy_(value)
        results.append(result)
    return results


def _empty_contiguous_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)


def _register_mamba3_op() -> Callable:
    global _MAMBA3_OP
    if _MAMBA3_OP is not None:
        return _MAMBA3_OP

    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        _Mamba3Function,
        mamba3_siso_combined,
    )

    def raw(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size):
        return mamba3_siso_combined(
            Q=q,
            K=k,
            V=v,
            ADT=adt,
            DT=dt,
            Trap=trap,
            Q_bias=q_bias,
            K_bias=k_bias,
            Angles=angles,
            D=d,
            Z=z,
            Input_States=None,
            chunk_size=chunk_size,
            return_final_states=False,
            cu_seqlens=None,
        )

    @torch.library.custom_op("atma::mamba3_siso_fwd", mutates_args=())
    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, adt: torch.Tensor,
                dt: torch.Tensor, trap: torch.Tensor, q_bias: torch.Tensor,
                k_bias: torch.Tensor, angles: torch.Tensor, d: torch.Tensor,
                z: torch.Tensor, chunk_size: int) -> torch.Tensor:
        return raw(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size)

    @forward.register_fake
    def _(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size):
        return torch.empty_like(v)

    @torch.library.custom_op("atma::mamba3_siso_bwd", mutates_args=())
    def backward(grad_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor, adt: torch.Tensor, dt: torch.Tensor,
                 trap: torch.Tensor, q_bias: torch.Tensor, k_bias: torch.Tensor,
                 angles: torch.Tensor, d: torch.Tensor, z: torch.Tensor,
                 chunk_size: int) -> list[torch.Tensor]:
        inputs = [q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z]

        # Nesting upstream Mamba inside torch.autograd.grad drops its rotary-angle
        # derivative. Drive the pinned Function's own methods so every fused-kernel
        # gradient, including dAngles, is preserved.
        class Context:
            needs_input_grad = (True,) * 18

            def save_for_backward(self, *tensors):
                self.saved_tensors = tensors

        ctx = Context()
        _Mamba3Function.forward(
            ctx,
            q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z,
            None, None, None, None, None, chunk_size, False,
        )
        gradients = _Mamba3Function.backward(ctx, grad_o)
        return _contiguous_gradients(inputs, gradients[:11])

    @backward.register_fake
    def _(grad_o, q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size):
        return [
            _empty_contiguous_like(tensor)
            for tensor in (q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z)
        ]

    def setup_context(ctx, inputs, output):
        *tensors, chunk_size = inputs
        ctx.save_for_backward(*tensors)
        ctx.chunk_size = chunk_size

    def autograd_backward(ctx, grad_o):
        gradients = backward(grad_o, *ctx.saved_tensors, ctx.chunk_size)
        return (*gradients, None)

    forward.register_autograd(autograd_backward, setup_context=setup_context)
    _MAMBA3_OP = forward
    return forward


def _register_mamba3_norm_op() -> Callable:
    global _MAMBA3_NORM_OP
    if _MAMBA3_NORM_OP is not None:
        return _MAMBA3_NORM_OP

    from fla.modules.layernorm_gated import rmsnorm_fn as upstream_rmsnorm_fn

    def raw(x, weight, eps):
        return upstream_rmsnorm_fn(
            x=x,
            weight=weight,
            bias=None,
            z=None,
            eps=eps,
            group_size=None,
            norm_before_gate=False,
        )

    @torch.library.custom_op("atma::mamba3_norm_fwd", mutates_args=())
    def forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return raw(x, weight, eps)

    @forward.register_fake
    def _(x, weight, eps):
        return torch.empty_like(x)

    @torch.library.custom_op("atma::mamba3_norm_bwd", mutates_args=())
    def backward(grad_y: torch.Tensor, x: torch.Tensor,
                 weight: torch.Tensor, eps: float) -> list[torch.Tensor]:
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(True) for tensor in (x, weight)]
            output = raw(*inputs, eps)
            gradients = torch.autograd.grad(output, inputs, grad_y)
            return _contiguous_gradients(inputs, gradients)

    @backward.register_fake
    def _(grad_y, x, weight, eps):
        return [_empty_contiguous_like(x), _empty_contiguous_like(weight)]

    def setup_context(ctx, inputs, output):
        x, weight, eps = inputs
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

    def autograd_backward(ctx, grad_y):
        gradients = backward(grad_y, *ctx.saved_tensors, ctx.eps)
        return (*gradients, None)

    forward.register_autograd(autograd_backward, setup_context=setup_context)
    _MAMBA3_NORM_OP = forward
    return forward


def install_external_custom_op(arch: str) -> str:
    """Patch Mamba at its fused, parameter-free kernel boundaries."""
    if arch in _PATCHED:
        return f"atma::{arch}"
    if arch != "mamba3_native":
        raise ValueError(f"no beneficial external custom op for {arch!r}")

    import fla.layers.mamba3 as layer_module
    norm_module = import_module("fla.modules.layernorm_gated")

    custom_op = _register_mamba3_op()
    norm_custom_op = _register_mamba3_norm_op()
    upstream = layer_module.mamba3_siso_combined
    upstream_norm = norm_module.rmsnorm_fn

    def wrapped_mamba3_siso_combined(*, Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
                                     Angles, D=None, Z=None, Input_States=None,
                                     chunk_size=64, return_final_states=False,
                                     cu_seqlens=None):
        supported = (
            D is not None
            and Z is not None
            and Input_States is None
            and not return_final_states
            and cu_seqlens is None
        )
        if supported:
            return custom_op(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, chunk_size)
        return upstream(
            Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
            Q_bias=Q_bias, K_bias=K_bias, Angles=Angles, D=D, Z=Z,
            Input_States=Input_States, chunk_size=chunk_size,
            return_final_states=return_final_states, cu_seqlens=cu_seqlens,
        )

    layer_module.mamba3_siso_combined = wrapped_mamba3_siso_combined

    def wrapped_rmsnorm_fn(x, weight, bias, z=None, eps=1e-6,
                           group_size=None, norm_before_gate=True):
        supported = (
            bias is None
            and z is None
            and group_size is None
            and not norm_before_gate
        )
        if supported:
            return norm_custom_op(x, weight, eps)
        return upstream_norm(
            x=x, weight=weight, bias=bias, z=z, eps=eps,
            group_size=group_size, norm_before_gate=norm_before_gate,
        )

    norm_module.rmsnorm_fn = wrapped_rmsnorm_fn
    _PATCHED.add(arch)
    return f"atma::{arch}"


def custom_op_is_installed(arch: str) -> bool:
    return arch in _PATCHED
