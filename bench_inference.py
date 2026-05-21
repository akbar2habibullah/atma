import time
import torch
from inference import LLM, SamplingParams

_WARMUP_PASSES = 3  # passes per batch size to let torch.compile finish before timing


def main():
    print("Initializing inference engine for performance benchmark...")
    llm = LLM(model="gpt2", kvcache_block_size=256)

    use_cuda = torch.cuda.is_available()

    # Benchmarking different batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    prompt = """Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.

Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.

Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.

Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.

Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos."""
    max_tokens = 256

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
            ignore_eos=True,  # force it to generate exactly max_tokens
        )

        # Multiple warmup passes to ensure torch.compile finishes compilation
        # before the timed run (first pass triggers JIT, subsequent passes verify it's done).
        for _ in range(_WARMUP_PASSES):
            llm.generate(prompts, sampling_params, use_tqdm=False)

        if use_cuda:
            torch.cuda.synchronize()

        # Timed benchmark pass
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_tokens = bs * max_tokens
        throughput = total_tokens / elapsed

        print(f"Batch Size: {bs:4d} | Generated: {total_tokens:5d} tokens | "
              f"Time: {elapsed:7.2f}s | Throughput: {throughput:7.2f} tok/s")


if __name__ == "__main__":
    main()
