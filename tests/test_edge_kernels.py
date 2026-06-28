import numpy as np
from tinygrad import Tensor, dtypes

from edge.kernels import gdn_prefill, gdn_prefill_chunked, polar_prefill
from edge.model import _gated_delta_sequential, _polar_reduce


def test_polar_prefill_matches_tensor_reference_fp32():
    Tensor.manual_seed(11)
    heads, tokens, head_dim, window = 2, 5, 4, 3
    q = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    k = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    v = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    v_null = Tensor.randn(heads, head_dim, dtype=dtypes.float32).realize()
    null_base = Tensor.randn(heads, dtype=dtypes.float32).realize()
    null_slope_raw = Tensor.randn(heads, dtype=dtypes.float32).realize()
    len_gain_raw = Tensor.randn(heads, dtype=dtypes.float32).realize()
    mag_beta_raw = Tensor.randn(heads, dtype=dtypes.float32).realize()

    q_b = q.reshape(1, heads, tokens, head_dim)
    k_b = k.reshape(1, heads, tokens, head_dim)
    v_b = v.reshape(1, heads, tokens, head_dim)
    key_idx = Tensor.arange(tokens).reshape(1, -1)
    query_next = Tensor.arange(1, tokens + 1, dtype=dtypes.float32)
    invalid = (key_idx >= query_next.reshape(-1, 1)) | (key_idx < (query_next.reshape(-1, 1) - window))
    sigma = (q_b @ k_b.transpose(-2, -1)) / (head_dim ** 0.5)
    sigma = sigma.masked_fill(invalid.reshape(1, 1, tokens, tokens), float("-inf"))
    expected_c, expected_m = _polar_reduce(
        sigma,
        v_b,
        query_next.minimum(float(window)),
        v_null=v_null,
        null_base=null_base,
        null_slope_raw=null_slope_raw,
        len_gain_raw=len_gain_raw,
        mag_beta_raw=mag_beta_raw,
    )

    got_c, got_m = polar_prefill(
        q,
        k,
        v,
        window_size=window,
        v_null=v_null,
        null_base=null_base,
        null_slope_raw=null_slope_raw,
        len_gain_raw=len_gain_raw,
        mag_beta_raw=mag_beta_raw,
    )
    assert np.allclose(got_c.numpy(), expected_c.reshape(heads, tokens, head_dim).numpy(), atol=1e-5, rtol=1e-5)
    assert np.allclose(got_m.numpy(), expected_m.reshape(heads, tokens).numpy(), atol=1e-5, rtol=1e-5)


def test_gdn_prefill_matches_tensor_reference_fp32():
    Tensor.manual_seed(12)
    heads, tokens, head_dim = 2, 5, 4
    q = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    k = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    v = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    gamma = Tensor.rand(heads, tokens, dtype=dtypes.float32).realize()
    beta = Tensor.rand(heads, tokens, dtype=dtypes.float32).realize()
    state = Tensor.randn(heads, head_dim, head_dim, dtype=dtypes.float32).realize()

    expected_read, expected_state = _gated_delta_sequential(
        q.reshape(1, heads, tokens, head_dim),
        k.reshape(1, heads, tokens, head_dim),
        v.reshape(1, heads, tokens, head_dim),
        gamma.reshape(1, heads, tokens),
        beta.reshape(1, heads, tokens),
        state.reshape(1, heads, head_dim, head_dim),
    )
    got_read, got_state = gdn_prefill(q, k, v, gamma, beta, state)
    assert np.allclose(got_read.numpy(), expected_read.reshape(heads, tokens, head_dim).numpy(), atol=1e-5, rtol=1e-5)
    assert np.allclose(got_state.numpy(), expected_state.reshape(heads, head_dim, head_dim).numpy(), atol=1e-5, rtol=1e-5)


def test_gdn_prefill_chunked_matches_tensor_reference_fp32():
    Tensor.manual_seed(13)
    heads, tokens, head_dim = 2, 9, 4
    q = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    k = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    v = Tensor.randn(heads, tokens, head_dim, dtype=dtypes.float32).realize()
    gamma = Tensor.rand(heads, tokens, dtype=dtypes.float32).realize()
    beta = Tensor.rand(heads, tokens, dtype=dtypes.float32).realize()
    state = Tensor.randn(heads, head_dim, head_dim, dtype=dtypes.float32).realize()

    expected_read, expected_state = _gated_delta_sequential(
        q.reshape(1, heads, tokens, head_dim),
        k.reshape(1, heads, tokens, head_dim),
        v.reshape(1, heads, tokens, head_dim),
        gamma.reshape(1, heads, tokens),
        beta.reshape(1, heads, tokens),
        state.reshape(1, heads, head_dim, head_dim),
    )
    got_read, got_state = gdn_prefill_chunked(q, k, v, gamma, beta, state, chunk_size=4)
    assert np.allclose(got_read.numpy(), expected_read.reshape(heads, tokens, head_dim).numpy(), atol=1e-5, rtol=1e-5)
    assert np.allclose(got_state.numpy(), expected_state.reshape(heads, head_dim, head_dim).numpy(), atol=1e-5, rtol=1e-5)
