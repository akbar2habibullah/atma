"""Illustrative single-GPU vLLM benchmark for Qwen3.5-9B.

This script intentionally uses token IDs and text-only mode so tokenizer and vision work do not
enter the comparison. Run it with the isolated vLLM environment documented in docs/kernel.md.
"""

import argparse
import json
import statistics
import time


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", help="comma-separated prompt lengths")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams, __version__ as vllm_version

    if args.lengths:
        lengths = [int(value) for value in args.lengths.split(",")]
    elif args.batch and args.prompt_length:
        lengths = [args.prompt_length] * args.batch
    else:
        parser.error("provide --lengths or both --batch and --prompt-length")
    batch, prompt_tokens = len(lengths), sum(lengths)
    max_model_len = args.max_model_len or max(lengths) + args.output_tokens
    max_num_batched_tokens = args.max_num_batched_tokens or max(8192, max(lengths))
    prompts = [{"prompt_token_ids": [1] * length} for length in lengths]
    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.output_tokens, ignore_eos=True)
    prefill_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=batch,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        seed=0,
    )

    def run_once(params):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        return elapsed, generated

    for _ in range(args.warmup):
        run_once(sampling)
    prefill_measured = [run_once(prefill_sampling) for _ in range(args.iterations)]
    if args.output_tokens == 1:
        measured = prefill_measured
    else:
        measured = [run_once(sampling) for _ in range(args.iterations)]
    prefill_elapsed = [item[0] for item in prefill_measured]
    elapsed = [item[0] for item in measured]
    generated = [item[1] for item in measured]
    prefill_p50 = _percentile(prefill_elapsed, 0.50)
    elapsed_p50 = _percentile(elapsed, 0.50)
    decode_seconds = elapsed_p50 - prefill_p50
    result = dict(
        model="Qwen/Qwen3.5-9B", framework="vLLM", vllm_version=vllm_version,
        dtype="bfloat16", tensor_parallel_size=1, language_model_only=True,
        prefix_caching=False, speculative_decoding=False,
        lengths=(lengths if len(lengths) <= 16 else None),
        length_min=min(lengths), length_max=max(lengths),
        length_mean=statistics.fmean(lengths),
        batch=batch, prompt_tokens=prompt_tokens,
        output_tokens_per_request=args.output_tokens,
        warmup=args.warmup, iterations=args.iterations,
        elapsed_p50_s=_percentile(elapsed, 0.50),
        elapsed_p95_s=_percentile(elapsed, 0.95),
        elapsed_mean_s=statistics.fmean(elapsed),
        prefill_elapsed_p50_s=prefill_p50,
        prompt_tok_s_median=prompt_tokens / prefill_p50,
        output_tok_s_median=statistics.median(
            count / seconds for count, seconds in zip(generated, elapsed)),
        steady_decode_tok_s_median=(
            batch * (args.output_tokens - 1) / decode_seconds
            if args.output_tokens > 1 and decode_seconds > 0 else None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    print("VLLM_QWEN_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
