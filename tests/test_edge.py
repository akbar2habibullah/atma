import torch
import pytest
import numpy as np
from tinygrad import dtypes
from tinygrad import Tensor, Variable

from edge.config import EdgeSamplingParams
from edge.engine import EdgeLLM
from edge.model import EdgeAtma
from model.config import AtmaConfig
from model.reference import ReferenceModel


def _tiny_config(mem_enabled=False, attn_window=None):
    return AtmaConfig(
        vocab_size=64,
        num_hidden_layers=4,
        hidden_size=32,
        head_dim=8,
        attn_kernel_size=3,
        conv_kernel_size=3,
        attn_window=attn_window,
        mem_enabled=mem_enabled,
        mem_chunk=4,
        dtype=torch.float32,
    )


@torch.inference_mode()
@pytest.mark.parametrize("mem_enabled", [False, True])
@pytest.mark.parametrize("attn_window", [None, 4])
def test_edge_matches_reference_full_prompt_fp32(mem_enabled, attn_window):
    torch.manual_seed(100 + int(mem_enabled) + (0 if attn_window is None else attn_window))
    cfg = _tiny_config(mem_enabled=mem_enabled, attn_window=attn_window)
    ref = ReferenceModel(cfg).eval()
    edge = EdgeAtma(cfg).to(device="CPU", dtype=dtypes.float32)
    edge.load_state_dict(ref.state_dict(), strict=True)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    got = edge(ids[0].tolist(), edge.new_state()).numpy()
    expected = ref(ids).numpy()
    assert np.allclose(got, expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("mem_enabled", [False, True])
@pytest.mark.parametrize("attn_window", [None, 4])
@pytest.mark.parametrize("splits", [[7], [3, 4], [1, 1, 1, 1, 1, 1, 1], [2, 1, 4]])
def test_edge_chunked_prompt_matches_reference_fp32(mem_enabled, attn_window, splits):
    torch.manual_seed(200 + int(mem_enabled) + (0 if attn_window is None else attn_window) + len(splits))
    cfg = _tiny_config(mem_enabled=mem_enabled, attn_window=attn_window)
    ref = ReferenceModel(cfg).eval()
    edge = EdgeAtma(cfg).to(device="CPU", dtype=dtypes.float32)
    edge.load_state_dict(ref.state_dict(), strict=True)

    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    expected = ref(ids).numpy()
    state = edge.new_state()
    chunks = []
    start = 0
    for width in splits:
        chunks.append(edge(ids[0, start:start + width].tolist(), state).numpy())
        start += width
    got = np.concatenate(chunks, axis=1)
    assert np.allclose(got, expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_edge_engine_generates_from_ids_with_random_weights():
    llm = EdgeLLM(model=None, device="cpu", dtype="fp32")
    out = llm.generate([1, 2], EdgeSamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))[0]
    assert len(out["token_ids"]) == 5


@torch.inference_mode()
def test_edge_static_decode_matches_dynamic_no_memory_fp32():
    torch.manual_seed(300)
    cfg = _tiny_config(mem_enabled=False, attn_window=4)
    edge = EdgeAtma(cfg).to(device="CPU", dtype=dtypes.float32)
    dyn_state = edge.new_state()
    static_state = edge.new_static_state(max_context=16)
    pos = Variable("static_pos", 0, 15)

    for i, token_id in enumerate([3, 1, 4, 1, 5, 9]):
        dynamic = edge([token_id], dyn_state).numpy()
        static = edge.decode_static(
            Tensor([[token_id]], device="CPU", dtype=dtypes.int32),
            pos.bind(i),
            static_state,
        ).numpy()
        assert np.allclose(static, dynamic, atol=5e-4, rtol=5e-4)


@torch.inference_mode()
def test_edge_static_decode_matches_dynamic_with_memory_fp32():
    torch.manual_seed(301)
    cfg = _tiny_config(mem_enabled=True, attn_window=4)
    edge = EdgeAtma(cfg).to(device="CPU", dtype=dtypes.float32)
    dyn_state = edge.new_state()
    static_state = edge.new_static_state(max_context=16)
    pos = Variable("static_mem_pos", 0, 15)

    for i, token_id in enumerate([3, 1, 4, 1, 5, 9]):
        dynamic = edge([token_id], dyn_state).numpy()
        static = edge.decode_static(
            Tensor([[token_id]], device="CPU", dtype=dtypes.int32),
            pos.bind(i),
            static_state,
        ).numpy()
        assert np.allclose(static, dynamic, atol=5e-4, rtol=5e-4)
