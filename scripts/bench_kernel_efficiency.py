"""Deterministic L40S gates for projection fusion and grouped prefill kernels."""

import argparse
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kernel.inference_ops_triton import linear_head_rms
from kernel.polar_triton import polar_attention_fwd, polar_attention_packed_fwd


DISTRIBUTIONS = {
    "short-heavy": [32, 48, 64, 64, 96, 128, 128, 256],
    "mixed": [64, 96, 128, 256, 512, 768, 1024, 1536],
    "long-tail": [64, 64, 128, 256, 512, 1024, 2048, 4096],
    "homogeneous": [512] * 8,
}


def samples_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    torch.cuda.reset_peak_memory_stats()
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    return {
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "mean": statistics.fmean(values),
        "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def print_result(name, result, tokens=None):
    throughput = "" if tokens is None else f" tok/s={tokens / (result['p50'] / 1e3):,.0f}"
    print(f"  {name:<10} p50={result['p50']:.4f} ms p95={result['p95']:.4f} ms "
          f"mean={result['mean']:.4f} ms peak={result['peak_mb']:.1f} MiB{throughput}")


def bench_a1(warmup, iterations):
    print("A1 projection + per-head RMSNorm (BF16, hidden=1024, head_dim=128)")
    for label, heads, gated in (("K", 2, False), ("Q", 8, True)):
        width = 128 * (2 if gated else 1)
        print(f" {label} projection")
        for rows in (1, 32, 256, 4096, 8192):
            x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(heads * width, 1024, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(heads * width, device="cuda", dtype=torch.bfloat16)

            def eager():
                y = F.linear(x, weight, bias).view(rows, heads, width)
                if gated:
                    q, gate = y.split(128, dim=-1)
                    return torch.cat((F.rms_norm(q, (128,)), gate), dim=-1)
                return F.rms_norm(y, (128,))

            def fused():
                return linear_head_rms(x, weight, bias, num_heads=heads, gated=gated)

            reference, actual = eager(), fused().view(rows, heads, width)
            error = (reference - actual).abs().max().item()
            base = samples_ms(eager, warmup, iterations)
            candidate = samples_ms(fused, warmup, iterations)
            speedup = base["p50"] / candidate["p50"]
            decision = "PASS" if speedup >= 1.10 else "REJECT"
            print(f"  M={rows:<4} eager={base['p50']:.4f} ms fused={candidate['p50']:.4f} ms "
                  f"speedup={speedup:.2f}x max_err={error:.5f} {decision}")


def tile_work(lengths, block_m=128, block_n=64, window=1024):
    pairs = 0
    for length in lengths:
        for query_start in range(0, length, block_m):
            hi = min(query_start + block_m, length)
            lo = max(0, hi - window) if window else 0
            pairs += (hi - lo + block_n - 1) // block_n
    return pairs


def bench_b(warmup, iterations):
    torch.manual_seed(0)
    heads, dim, window = 8, 128, 1024
    params = dict(
        v_null=torch.randn(heads, dim, device="cuda"),
        null_base=torch.randn(heads, device="cuda"),
        null_slope_raw=torch.randn(heads, device="cuda"),
        len_gain_raw=torch.randn(heads, device="cuda"),
        mag_beta_raw=torch.randn(heads, device="cuda"),
    )
    print("B1/B2 packed fresh heterogeneous Polar (BF16, H=8, D=128, window=1024)")
    for name, lengths in DISTRIBUTIONS.items():
        total = sum(lengths)
        q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        seq_starts, query_starts, seq_lens = [], [], []
        packed_start = 0
        for length in lengths:
            for query_start in range(0, length, 128):
                seq_starts.append(packed_start)
                query_starts.append(query_start)
                seq_lens.append(length)
            packed_start += length
        tile_map = [torch.tensor(x, device="cuda", dtype=torch.int32)
                    for x in (seq_starts, query_starts, seq_lens)]

        def oracle():
            outputs, start = [], 0
            for length in lengths:
                sl = slice(start, start + length)
                n_keys = torch.arange(1, length + 1, device="cuda", dtype=torch.float32)
                outputs.append(polar_attention_fwd(
                    q[sl].transpose(0, 1)[None].contiguous(),
                    k[sl].transpose(0, 1)[None].contiguous(),
                    v[sl].transpose(0, 1)[None].contiguous(), n_keys,
                    is_causal=True, window=window, **params))
                start += length
            return outputs

        def grouped():
            return polar_attention_packed_fwd(q, k, v, *tile_map, window=window, **params)

        # Compile, then verify exactly against the established route.
        grouped_c, grouped_mag = grouped()
        expected = oracle()
        expected_c = torch.cat([c[0].transpose(0, 1) for c, _ in expected])
        expected_mag = torch.cat([m[0].transpose(0, 1) for _, m in expected])
        max_error = max((grouped_c - expected_c).abs().max().item(),
                        (grouped_mag - expected_mag).abs().max().item())
        base = samples_ms(oracle, warmup, iterations)
        candidate = samples_ms(grouped, warmup, iterations)
        speedup = base["p50"] / candidate["p50"]
        decision = "PASS" if speedup >= 1.20 else "REJECT"
        print(f" {name}: lengths={lengths} tiles={len(seq_starts)} "
              f"effective_tile_pairs={tile_work(lengths)} max_err={max_error:.5f}")
        print_result("oracle", base, total)
        print_result("grouped", candidate, total)
        print(f"  speedup={speedup:.2f}x Polar launches={len(lengths)}->1 {decision}")


def bench_full_model(distribution, warmup, iterations):
    """One distribution per process keeps FLA's shape caches out of peak-memory results."""
    from inference.models.atma import Atma
    from inference.utils.context import get_context, reset_context, set_context
    from model.config import AtmaConfig

    lengths = DISTRIBUTIONS[distribution]
    batch, total = len(lengths), sum(lengths)
    old_device, old_dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    config = AtmaConfig()
    model = Atma(config).eval()
    torch.set_default_device(old_device)
    torch.set_default_dtype(old_dtype)
    # Deterministic finite weights; values do not alter the executed GEMM/kernel shapes.
    for parameter in model.parameters():
        parameter.data.zero_()

    num_blocks = (total + 255) // 256
    for module in model.modules():
        if hasattr(module, "k_cache"):
            module.k_cache = torch.zeros(
                num_blocks, 256, config.num_key_value_heads, config.head_dim,
                device="cuda", dtype=config.dtype)
            module.v_cache = torch.zeros_like(module.k_cache)
    tables = {}
    for layer in range(config.num_hidden_layers):
        if layer % 4 == 2:
            for suffix in "qkv":
                channels = (config.hidden_size if suffix == "q"
                            else config.num_key_value_heads * config.head_dim)
                tables[f"attn_{layer}_{suffix}"] = torch.zeros(
                    batch, channels, config.attn_kernel_size - 1,
                    device="cuda", dtype=config.dtype)
            tables[f"mem_{layer}"] = torch.zeros(
                batch, config.num_attention_heads, config.head_dim, config.head_dim,
                device="cuda")
        else:
            tables[f"conv_{layer}_gated"] = torch.zeros(
                batch, config.hidden_size, config.conv_kernel_size - 1,
                device="cuda", dtype=config.dtype)
    seqs = [SimpleNamespace(num_cached_tokens=0, seq_slot=i, block_table=[])
            for i in range(batch)]
    ids = torch.zeros(total, device="cuda", dtype=torch.long)

    def set_route(grouped):
        kwargs = {}
        if grouped:
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
            values = [torch.tensor(v, device="cuda", dtype=torch.int32) for v in (
                seq_starts, query_starts, seq_lens,
                token_starts, token_ends, token_slots)]
            kwargs = dict(
                grouped_polar_prefill=True,
                polar_tile_seq_starts=values[0], polar_tile_q_starts=values[1],
                polar_tile_seq_lens=values[2], token_seq_starts=values[3],
                token_seq_ends=values[4], token_seq_slots=values[5])
        set_context(
            True, seqlens_q=lengths,
            slot_mapping=torch.arange(total, device="cuda", dtype=torch.int32),
            conv_state_tables=tables, **kwargs)
        get_context().seqs = seqs

    results = {}
    with torch.inference_mode():
        for route, grouped in (("oracle", False), ("grouped", True)):
            set_route(grouped)
            results[route] = samples_ms(lambda: model(ids), warmup, iterations)
    reset_context()
    print(f"Full 16-layer model: {distribution} lengths={lengths} random_status=zero BF16 TP=1")
    print_result("oracle", results["oracle"], total)
    print_result("grouped", results["grouped"], total)
    speedup = results["oracle"]["p50"] / results["grouped"]["p50"]
    print(f"  speedup={speedup:.2f}x {'PASS' if speedup >= 1.20 else 'REJECT'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--only", choices=("a1", "b"))
    parser.add_argument("--full-model", choices=tuple(DISTRIBUTIONS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} CUDA={torch.version.cuda}")
    if args.full_model:
        bench_full_model(args.full_model, args.warmup, args.iterations)
        return
    if args.only in (None, "a1"):
        bench_a1(args.warmup, args.iterations)
    if args.only in (None, "b"):
        bench_b(args.warmup, args.iterations)


if __name__ == "__main__":
    main()
