from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Device, dtypes


_DTYPES = {
    "auto": None,
    "fp16": dtypes.float16,
    "float16": dtypes.float16,
    "bf16": dtypes.bfloat16,
    "bfloat16": dtypes.bfloat16,
    "fp32": dtypes.float32,
    "float32": dtypes.float32,
}

_DEVICES = {
    "cpu": "CPU",
    "opencl": "CL",
    "cl": "CL",
    "gpu": "GPU",
    "cuda": "CUDA",
    "webgpu": "WEBGPU",
    "amd": "AMD",
}


@dataclass
class EdgeConfig:
    """Runtime knobs for the small edge engine.

    ``backend`` is explicit for future native/runtime variants.  The first
    implementation is tinygrad-native and can run on tinygrad devices such as
    CPU, CL/OpenCL, and WEBGPU when those backends are available.
    """

    model: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    backend: str = "tinygrad"
    max_context: int | None = None
    tokenizer: str | None = None
    eos_token_id: int = 50256
    use_checkpoint_config: bool = True


@dataclass
class EdgeSamplingParams:
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50
    eos_token_id: int | None = None
    ignore_eos: bool = False


def resolve_device(device: str) -> str:
    if device == "auto":
        return Device.DEFAULT
    return _DEVICES.get(device.lower(), device.upper())


def resolve_dtype(dtype: str, device: str):
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype '{dtype}', expected one of {sorted(_DTYPES)}")
    if dtype == "auto":
        return dtypes.float32 if device == "CPU" else dtypes.float16
    return _DTYPES[dtype]
