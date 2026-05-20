from inference import LLM, SamplingParams


def main():
    print("Initializing inference engine for Atma codebase...")
    
    # We use 'gpt2' to load standard fast tokenizer settings.
    # Since no trained weights file is specified, it will initialize Atma
    # with random weights to demonstrate inference execution.
    llm = LLM(model="gpt2", kvcache_block_size=16)

    prompts = [
        "In the heart of artificial intelligence research, a new model called Atma was born. It used",
        "Liquid Foundation Models (LFM2) leverage gated causal convolutions to",
        "The main difference between standard attention and paged attention is that paged attention",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        ignore_eos=False,
    )

    print("\nStarting autoregressive text generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 80)
    print("GENERATED COMPLETIONS")
    print("=" * 80)
    for prompt, out in zip(prompts, outputs):
        print(f"\n[PROMPT]    : {prompt}")
        print(f"[GENERATED] : {out['text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
