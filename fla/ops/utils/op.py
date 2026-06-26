"""Triton math wrappers used by the Wall kernels."""

import triton
import triton.language as tl


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def log2(x):
    return tl.log2(x.to(tl.float32))
