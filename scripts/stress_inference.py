"""Stress the paged inference model at configurable architecture and workload shapes.

Run one workload per process so CUDA/FLA caches do not contaminate peak-memory results.
Weights are deterministic zeros unless a checkpoint loader is added; kernel/GEMM shapes and
memory traffic are identical to populated BF16 weights.
"""

import argparse
import json
import math
import statistics
from types import SimpleNamespace

import torch


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def _measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _make_model(args):
    from inference.models.atma import Atma
    from model.config import AtmaConfig

    config = AtmaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        head_dim=args.head_dim,
        attn_window=args.window,
        mem_enabled=True,
        dtype=torch.bfloat16,
    )
    old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Atma(config).eval()
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model, config


def _allocate_states(config, batch):
    tables = {}
    kv_hidden = config.num_key_value_heads * config.head_dim
    for layer in range(config.num_hidden_layers):
        if layer % 4 == 2:
            tables[f"attn_{layer}_q"] = torch.zeros(
                batch, config.hidden_size, config.attn_kernel_size - 1,
                device="cuda", dtype=config.dtype)
            for suffix in ("k", "v"):
                tables[f"attn_{layer}_{suffix}"] = torch.zeros(
                    batch, kv_hidden, config.attn_kernel_size - 1,
                    device="cuda", dtype=config.dtype)
            tables[f"mem_{layer}"] = torch.zeros(
                batch, config.num_attention_heads, config.head_dim, config.head_dim,
                device="cuda", dtype=torch.float32)
        else:
            tables[f"conv_{layer}_gated"] = torch.zeros(
                batch, config.hidden_size, config.conv_kernel_size - 1,
                device="cuda", dtype=config.dtype)
    return tables


def _attach_cache(model, config, num_blocks, block_size):
    caches = []
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = torch.zeros(
                num_blocks, block_size, config.num_key_value_heads, config.head_dim,
                device="cuda", dtype=config.dtype)
            module.v_cache = torch.zeros_like(module.k_cache)
            caches.extend((module.k_cache, module.v_cache))
    return caches


def _prefill_context(lengths, tables, route, block_size):
    from inference.utils.context import get_context, set_context

    batch, total = len(lengths), sum(lengths)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    kwargs = {}
    if route == "dense":
        kwargs.update(
            dense_prefill=True, dense_batch_size=batch, dense_seq_len=lengths[0],
            seq_slots=torch.arange(batch, device="cuda", dtype=torch.int64))
    elif route == "grouped":
        seq_starts, query_starts, seq_lens = [], [], []
        token_starts, token_ends, token_slots = [], [], []
        packed_start = 0
        for slot, length in enumerate(lengths):
            for query_start in range(0, length, 128):
                seq_starts.append(packed_start)
                query_starts.append(query_start)
                seq_lens.append(length)
            token_starts.extend([packed_start] * length)
            token_ends.extend([packed_start + length] * length)
            token_slots.extend([slot] * length)
            packed_start += length
        values = [torch.tensor(value, device="cuda", dtype=torch.int32) for value in (
            seq_starts, query_starts, seq_lens, token_starts, token_ends, token_slots)]
        kwargs.update(
            grouped_polar_prefill=True,
            polar_tile_seq_starts=values[0], polar_tile_q_starts=values[1],
            polar_tile_seq_lens=values[2], token_seq_starts=values[3],
            token_seq_ends=values[4], token_seq_slots=values[5],
            seq_slots=torch.arange(batch, device="cuda", dtype=torch.int64))
    set_context(
        True,
        cu_seqlens_q=torch.tensor(cumulative, device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor(cumulative, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
        seqlens_q=lengths,
        slot_mapping=torch.arange(total, device="cuda", dtype=torch.int32),
        conv_state_tables=tables,
        **kwargs,
    )
    get_context().seqs = [
        SimpleNamespace(num_cached_tokens=0, seq_slot=slot, block_table=[])
        for slot in range(batch)
    ]


def _run_prefill(args, model, config):
    from inference.utils.context import reset_context
    from inference.layers.sampler import Sampler

    if args.lengths:
        lengths = [int(value) for value in args.lengths.split(",")]
    else:
        lengths = [args.prompt_length] * args.batch
    homogeneous = len(set(lengths)) == 1
    route = "dense" if homogeneous and len(lengths) > 1 else (
        "grouped" if len(lengths) > 1 else "oracle")
    total = sum(lengths)
    tables = _allocate_states(config, len(lengths))
    num_blocks = math.ceil(total / args.block_size)
    _attach_cache(model, config, num_blocks, args.block_size)
    _prefill_context(lengths, tables, route, args.block_size)
    ids = torch.zeros(total, device="cuda", dtype=torch.int64)
    sampler = Sampler() if args.include_sampler else None
    temperatures = torch.ones(len(lengths), device="cuda", dtype=torch.float32)

    @torch.inference_mode()
    def step():
        logits = model.compute_logits(model(ids))
        return sampler(logits, temperatures) if sampler is not None else logits

    result = _measure(step, args.warmup, args.iterations)
    result.update(
        mode="prefill", route=route, batch=len(lengths), lengths=lengths,
        tokens=total, throughput_tok_s=total / (result["p50_ms"] / 1000.0),
        includes_sampler=args.include_sampler)
    reset_context()
    return result


def _decode_context(context_lengths, tables, block_size):
    from inference.utils.context import set_context

    batch = len(context_lengths)
    block_counts = [math.ceil(length / block_size) for length in context_lengths]
    max_blocks = max(block_counts)
    rows, last_blocks, next_block = [], [], 0
    for count in block_counts:
        live = list(range(next_block, next_block + count))
        rows.append(live + [-1] * (max_blocks - count))
        last_blocks.append(live[-1])
        next_block += count
    block_tables = torch.tensor(rows, device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor(last_blocks, device="cuda", dtype=torch.int32) * block_size
    slot_mapping += torch.tensor(
        [(length - 1) % block_size for length in context_lengths],
        device="cuda", dtype=torch.int32)
    set_context(
        False,
        slot_mapping=slot_mapping.to(torch.int32),
        context_lens=torch.tensor(context_lengths, device="cuda", dtype=torch.int32),
        block_tables=block_tables,
        seq_slots=torch.arange(batch, device="cuda", dtype=torch.int64),
        conv_state_tables=tables,
    )
    return next_block


def _run_decode(args, model, config):
    from inference.utils.context import reset_context
    from inference.layers.sampler import Sampler

    if args.context_lengths:
        pattern = [int(value) for value in args.context_lengths.split(",")]
        context_lengths = pattern * args.repeat
    else:
        context_lengths = [args.context_length] * args.batch
    batch = len(context_lengths)
    tables = _allocate_states(config, batch)
    total_blocks = sum(math.ceil(length / args.block_size) for length in context_lengths)
    _attach_cache(model, config, total_blocks, args.block_size)
    _decode_context(context_lengths, tables, args.block_size)
    ids = torch.zeros(batch, device="cuda", dtype=torch.int64)
    static_hidden = torch.empty(
        batch, config.hidden_size, device="cuda", dtype=config.dtype)
    sampler = Sampler() if args.include_sampler else None
    temperatures = torch.ones(batch, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        static_hidden.copy_(model(ids))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_hidden.copy_(model(ids))

    @torch.inference_mode()
    def step():
        graph.replay()
        logits = model.compute_logits(static_hidden)
        return sampler(logits, temperatures) if sampler is not None else logits

    result = _measure(step, args.warmup, args.iterations)
    result.update(
        mode="decode", route="cuda_graph", batch=batch,
        context_length=(context_lengths[0] if len(set(context_lengths)) == 1 else None),
        context_min=min(context_lengths), context_max=max(context_lengths),
        context_mean=statistics.fmean(context_lengths), tokens=batch,
        throughput_tok_s=batch / (result["p50_ms"] / 1000.0),
        includes_sampler=args.include_sampler)
    reset_context()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--lengths", help="comma-separated heterogeneous prefill lengths")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--context-lengths", help="comma-separated heterogeneous decode contexts")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat --context-lengths pattern to form a larger batch")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--include-sampler", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.hidden_size % args.head_dim:
        raise SystemExit("hidden size must be divisible by head dim")

    torch.manual_seed(0)
    model, config = _make_model(args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    try:
        result = (_run_prefill(args, model, config) if args.mode == "prefill"
                  else _run_decode(args, model, config))
        result.update(
            status="ok", gpu=torch.cuda.get_device_name(), dtype="bfloat16",
            hidden_size=args.hidden_size, layers=args.layers,
            parameter_count=parameter_count,
            parameter_gib=parameter_count * 2 / 2**30,
            warmup=args.warmup, iterations=args.iterations,
        )
    except torch.OutOfMemoryError as error:
        attempted_batch = (len(args.context_lengths.split(",")) * args.repeat
                           if args.mode == "decode" and args.context_lengths else args.batch)
        result = dict(
            status="oom", mode=args.mode, batch=attempted_batch,
            hidden_size=args.hidden_size, layers=args.layers,
            parameter_count=parameter_count,
            allocated_gib=torch.cuda.memory_allocated() / 2**30,
            reserved_gib=torch.cuda.memory_reserved() / 2**30,
            error=str(error).splitlines()[0],
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
