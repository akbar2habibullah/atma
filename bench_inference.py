import time
from inference import LLM, SamplingParams


def main():
    print("Initializing inference engine for performance benchmark...")
    llm = LLM(model="gpt2", kvcache_block_size=16)

    # Benchmarking different batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64]
    prompt = "Artificial Intelligence and Machine Learning are transforming modern technology by enabling computers to"
    max_tokens = 50

    print(f"\nBenchmark Configuration:")
    print(f"- Model Architecture: Atma ({llm.engine.config.hf_config.num_hidden_layers} layers, {llm.engine.config.hf_config.hidden_size} model dim, causal conv + attention + LFM2)")
    print(f"- Max Tokens to Generate: {max_tokens} per sequence")
    print(f"- Prompt Length: {len(prompt.split())} words")

    print("\nRunning benchmarks...")
    for bs in batch_sizes:
        prompts = [prompt] * bs
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_tokens,
            ignore_eos=True, # force it to generate exactly max_tokens
        )

        # Warmup pass
        llm.generate(prompts, sampling_params, use_tqdm=False)

        # Timed benchmark pass
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        total_tokens = bs * max_tokens
        throughput = total_tokens / elapsed

        print(f"Batch Size: {bs:2d} | Generated: {total_tokens:3d} tokens | "
              f"Time: {elapsed:5.2f}s | Throughput: {throughput:6.2f} tok/s")


if __name__ == "__main__":
    main()
