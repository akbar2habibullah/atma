from __future__ import annotations

import argparse

from edge.config import EdgeConfig, EdgeSamplingParams
from edge.engine import EdgeLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Atma edge inference runtime")
    parser.add_argument("--model", "-m", default=None, help="checkpoint file or directory")
    parser.add_argument("--prompt", default=None, help="text prompt; requires tiktoken or transformers")
    parser.add_argument("--ids", type=int, nargs="*", default=None, help="raw token ids prompt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cl/opencl, webgpu, cuda, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    llm = EdgeLLM(EdgeConfig(model=args.model, device=args.device, dtype=args.dtype))
    print(f"[edge] loaded={llm.info['loaded']} path={llm.info['path']}")
    print(f"[edge] backend=tinygrad device={llm.info['device']} dtype={llm.info['dtype']} tokenizer={llm.info['tokenizer']}")
    if not llm.info["loaded"]:
        print("[edge] no checkpoint found; running random weights")

    if args.ids is not None and len(args.ids) > 0:
        prompt = args.ids
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = "The"

    sampling = EdgeSamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        ignore_eos=args.ignore_eos,
    )
    out = llm.generate(prompt, sampling)[0]
    print("\n[token ids]", out["token_ids"])
    if out["text"]:
        print("\n[text]\n" + out["text"])


if __name__ == "__main__":
    main()
