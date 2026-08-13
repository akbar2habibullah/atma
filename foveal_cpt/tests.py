"""Portable correctness checks for the Foveal pilot.

Run with the same Python environment used for training:

    python -m foveal_cpt.tests
"""

from __future__ import annotations

import copy
import struct
import tempfile
from pathlib import Path

import torch

from kernel.polar_triton import HAS_TRITON, polar_attention_sparse
from model.blocks import polar_reduce
from model.config import AtmaConfig
from train.model import Model

from .attention import FovealAttention, select_pages
from .checkpoint import wrap_foveal
from .config import FovealConfig
from .model import FovealCPTModel, foveal_layers
from .prepare_data import ensure_training_data, shard_token_count


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}" + (f" ({detail})" if detail else ""))


def tiny_config(attn_type: str, *, cuda_flex: bool = False) -> tuple[AtmaConfig, FovealConfig]:
    sequence_length = 128 if cuda_flex else 32
    hidden_size = 128 if cuda_flex else 32
    head_dim = 32 if cuda_flex else 8
    page_size = 64 if cuda_flex else 8
    atma = AtmaConfig(
        vocab_size=64,
        num_hidden_layers=4,
        hidden_size=hidden_size,
        head_dim=head_dim,
        max_position_embeddings=sequence_length,
        attn_type=attn_type,
        attn_kernel="torch",
        attn_window=None,
        mem_enabled=False,
    )
    foveal = FovealConfig(
        checkpoint="unused",
        adaptation_mode="local",
        sequence_length=sequence_length,
        batch_tokens=sequence_length,
        microbatch_sequences=1,
        train_tokens=32,
        index_dim=8,
        page_size=page_size,
        query_block_size=page_size,
        local_window=sequence_length,
        remote_capacity=4,
        top_p=0.95,
        min_remote_pages=0,
        max_remote_pages=4,
        initial_min_remote_pages=0,
        initial_max_remote_pages=4,
        teacher_query_blocks=2,
        activation_checkpointing=False,
        flex_compile=cuda_flex,
        flex_kernel_options=(
            {"BLOCK_M": 64, "BLOCK_N": 64, "PRESCALE_QK": True}
            if cuda_flex
            else None
        ),
        xent_impl="chunked",
    )
    foveal.validate()
    return atma, foveal


def test_router() -> None:
    scores = torch.zeros(1, 4, 4)
    scores[0, 3, 0] = 5.0
    route = select_pages(
        scores,
        page_size=8,
        local_window=8,
        top_p=0.9,
        min_remote_pages=0,
        max_remote_pages=2,
        remote_capacity=2,
    )
    check("router has no remote page at block zero", int(route.page_counts[0, 0]) == 0)
    check("router selects causal page", int(route.page_indices[0, 3, 0]) == 0)
    check("router respects cap", int(route.page_counts.max()) <= 2)


def test_dataset_preflight() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "finewebedu_train_000001.bin"
        token_count = 10
        header = [20240520, 1, token_count, 2] + [0] * 252
        path.write_bytes(struct.pack("<256i", *header) + bytes(token_count * 2))
        config = FovealConfig(
            train_glob=str(Path(directory) / "finewebedu_train_*.bin"),
            dataset_repo=None,
            auto_download_data=False,
            sequence_length=8,
            batch_tokens=8,
            microbatch_sequences=1,
            train_tokens=8,
            index_dim=8,
            page_size=8,
            query_block_size=8,
            local_window=8,
            remote_capacity=4,
            max_remote_pages=4,
            initial_max_remote_pages=4,
        )
        paths = ensure_training_data(config, include_validation=False)
        check("dataset preflight finds the sequential shard", paths == [path])
        check("dataset preflight validates token count", shard_token_count(path) == token_count)


def test_dense_parity(attn_type: str, device: str = "cpu", *, backward: bool = False) -> None:
    torch.manual_seed(7)
    cuda_flex = device == "cuda"
    atma, config = tiny_config(attn_type, cuda_flex=cuda_flex)
    dense = Model(atma, reg_mode="baseline").to(device).eval()
    sparse_base = copy.deepcopy(dense)
    wrap_foveal(sparse_base, config)
    sparse_base.to(device)
    sparse = FovealCPTModel(sparse_base, config).eval()
    sparse.set_mode("sparse")
    for layer in foveal_layers(sparse.base):
        layer.teacher_query_blocks = 0
    inputs = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int32, device=device)
    targets = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int64, device=device)
    with torch.no_grad():
        dense_loss = dense(inputs, targets)[0]
    sparse_loss = sparse(inputs, targets)[0]
    difference = abs(float(dense_loss.detach()) - float(sparse_loss.detach()))
    # CPU SDPA and the explicit boolean-mask reference use different reduction
    # orders; this is a summed-token loss, so keep the per-token tolerance tight.
    tolerance = 1e-1 if cuda_flex else 1e-2
    check(
        f"{device} {attn_type} full-support loss parity",
        difference < tolerance,
        f"difference={difference:.3e}",
    )
    if backward:
        sparse_loss.backward()
        grads = [parameter.grad for parameter in sparse.parameters() if parameter.requires_grad]
        check(f"{device} {attn_type} backward produced gradients", any(grad is not None for grad in grads))
        check(
            f"{device} {attn_type} backward gradients are finite",
            all(torch.isfinite(grad).all() for grad in grads if grad is not None),
        )


def test_index_gradient() -> None:
    torch.manual_seed(11)
    atma, config = tiny_config("polar")
    model = Model(atma, reg_mode="baseline")
    wrap_foveal(model, config)
    wrapped = FovealCPTModel(model, config)
    wrapped.set_mode("dense_teacher")
    wrapped.freeze_except_index()
    inputs = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int32)
    loss = wrapped.calibration_loss(inputs)
    loss.backward()
    grads = [parameter.grad for parameter in wrapped.routing_parameters()]
    check("index calibration produces gradients", all(grad is not None for grad in grads))
    check("index gradients are finite", all(torch.isfinite(grad).all() for grad in grads if grad is not None))


def test_lm_output_gradient() -> None:
    torch.manual_seed(13)
    atma, config = tiny_config("polar")
    config.adaptation_mode = "lm_output"
    model = Model(atma, reg_mode="baseline")
    wrap_foveal(model, config)
    wrapped = FovealCPTModel(model, config)
    wrapped.configure_adaptation()
    inputs = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int32)
    targets = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int64)
    lm_loss, _, index_loss = wrapped(inputs, targets)
    check("LM-output cell has no KL loss", float(index_loss.detach()) == 0.0)
    lm_loss.backward()
    for layer in foveal_layers(wrapped.base):
        for name, module in (
            ("q", layer.index_q),
            ("k", layer.index_k),
            ("v", layer.index_v),
            ("out", layer.index_out),
        ):
            grads = [parameter.grad for parameter in module.parameters()]
            check(f"LM loss reaches index {name}", all(grad is not None for grad in grads))
            check(
                f"LM index {name} gradients are finite",
                all(torch.isfinite(grad).all() for grad in grads if grad is not None),
            )


def test_sparse_kl_gradient() -> None:
    torch.manual_seed(17)
    atma, config = tiny_config("polar")
    config.adaptation_mode = "kl"
    model = Model(atma, reg_mode="baseline")
    wrap_foveal(model, config)
    wrapped = FovealCPTModel(model, config)
    wrapped.configure_adaptation()
    inputs = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int32)
    targets = torch.randint(0, atma.vocab_size, (1, config.sequence_length), dtype=torch.int64)
    lm_loss, _, index_loss = wrapped(inputs, targets)
    check("selected-support KL is finite", torch.isfinite(index_loss).item())
    (lm_loss + index_loss).backward()
    grads = [parameter.grad for parameter in wrapped.routing_parameters()]
    check("selected-support KL reaches routing projections", all(grad is not None for grad in grads))
    check(
        "selected-support routing gradients are finite",
        all(torch.isfinite(grad).all() for grad in grads if grad is not None),
    )


def test_sparse_polar_triton() -> None:
    """Compare selected-page forward and backward with the materialized oracle."""
    if not torch.cuda.is_available() or not HAS_TRITON:
        return
    torch.manual_seed(23)
    device = "cuda"
    batch, heads, tokens, dim = 1, 2, 64, 16
    page_size, local_window, capacity = 16, 16, 2
    pages = tokens // page_size
    page_indices = torch.zeros((batch, pages, capacity), device=device, dtype=torch.int32)
    page_counts = torch.tensor([[0, 0, 1, 2]], device=device, dtype=torch.int32)
    page_indices[0, 2, 0] = 0
    page_indices[0, 3, 0] = 0
    page_indices[0, 3, 1] = 1

    def tensor(shape):
        return torch.randn(shape, device=device, dtype=torch.float32) * 0.2

    q0, k0, v0 = tensor((batch, heads, tokens, dim)), tensor((batch, heads, tokens, dim)), tensor((batch, heads, tokens, dim))
    params0 = [
        tensor((heads, dim)),
        tensor((heads,)),
        tensor((heads,)),
        tensor((heads,)),
        tensor((heads,)),
    ]
    sparse_inputs = [value.detach().clone().requires_grad_() for value in (q0, k0, v0, *params0)]
    oracle_inputs = [value.detach().clone().requires_grad_() for value in (q0, k0, v0, *params0)]
    sq, sk, sv, svn, snb, sns, slg, smb = sparse_inputs
    oq, ok, ov, ovn, onb, ons, olg, omb = oracle_inputs

    actual_c, actual_mag = polar_attention_sparse(
        sq, sk, sv, page_indices, page_counts,
        page_size=page_size, local_window=local_window,
        v_null=svn, null_base=snb, null_slope_raw=sns,
        len_gain_raw=slg, mag_beta_raw=smb,
    )
    positions = torch.arange(tokens, device=device)
    allowed = ((positions[None, :] <= positions[:, None])
               & (positions[None, :] > positions[:, None] - local_window))
    allowed = allowed.unsqueeze(0)
    for query_page in range(pages):
        q0_idx, q1_idx = query_page * page_size, (query_page + 1) * page_size
        for slot in range(int(page_counts[0, query_page])):
            key_page = int(page_indices[0, query_page, slot])
            k0_idx, k1_idx = key_page * page_size, (key_page + 1) * page_size
            allowed[0, q0_idx:q1_idx, k0_idx:k1_idx] = True
    scores = torch.matmul(oq, ok.transpose(-2, -1)) / (dim ** 0.5)
    scores = scores.masked_fill(~allowed[:, None], -torch.inf)
    support = (positions + 1).float()
    expected_c, expected_mag = polar_reduce(
        scores, ov, support,
        v_null=ovn, null_base=onb, null_slope_raw=ons,
        len_gain_raw=olg, mag_beta_raw=omb,
    )
    torch.testing.assert_close(actual_c, expected_c, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_mag, expected_mag, atol=2e-4, rtol=2e-4)
    check("CUDA sparse Polar Triton forward matches oracle", True)

    grad_c, grad_mag = torch.randn_like(actual_c), torch.randn_like(actual_mag)
    torch.autograd.backward((actual_c, actual_mag), (grad_c, grad_mag))
    torch.autograd.backward((expected_c, expected_mag), (grad_c, grad_mag))
    for name, actual, expected in zip(
        ("q", "k", "v", "v_null", "null_base", "null_slope", "len_gain", "mag_beta"),
        sparse_inputs,
        oracle_inputs,
    ):
        torch.testing.assert_close(actual.grad, expected.grad, atol=3e-3, rtol=3e-3)
        check(f"CUDA sparse Polar Triton {name} gradient matches oracle", True)


def main() -> None:
    test_dataset_preflight()
    test_router()
    test_dense_parity("nope")
    test_dense_parity("rope")
    test_dense_parity("polar")
    test_index_gradient()
    test_lm_output_gradient()
    test_sparse_kl_gradient()
    if torch.cuda.is_available():
        test_sparse_polar_triton()
        test_dense_parity("nope", "cuda", backward=True)
        test_dense_parity("polar", "cuda", backward=True)
    print("All Foveal CPT tests passed.")


if __name__ == "__main__":
    main()
