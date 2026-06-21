"""EvalModel: thin adapter over the production inference engine (inference.LLM) exposing the
autoregressive generate() interface the benchmark harness needs.

>>> WIP DISCLAIMER <<<
The paged inference engine (inference/) currently runs LEGACY softmax causal attention only.
It does NOT yet implement: Polar attention, the Titans memory branch, the Canon convs, or the
training sliding window (see docs/inference.md — porting these is the tracked future task).
Therefore, for ANY checkpoint trained in the polar/titans line, this engine's generations are
NUMERICALLY INVALID, and every benchmark score produced through this adapter is a PLACEHOLDER
until the inference port is complete. This module is written now against the intended stable
interface so the harness runs unchanged once inference lands.
"""
import json
import os

_BANNER = "=" * 78


def read_checkpoint_config(model_path: str) -> dict:
    """Load the AtmaConfig JSON saved alongside a checkpoint (config.json)."""
    p = model_path
    if os.path.isfile(p):
        p = os.path.dirname(p)
    cfg_path = os.path.join(p, "config.json")
    if os.path.exists(cfg_path):
        try:
            return json.load(open(cfg_path))
        except Exception:
            return {}
    return {}


def unsupported_features(cfg: dict) -> list:
    """Features present in a checkpoint that the paged inference engine does NOT yet run."""
    missing = []
    if cfg.get("attn_type") == "polar":
        missing.append("Polar attention core (engine runs legacy softmax)")
    if cfg.get("attn_type") in ("polar", "nope"):
        missing.append("Canon convs on Q/K/V")
    if cfg.get("mem_enabled"):
        missing.append("Titans memory branch (no recurrent memory state in the paged engine)")
    if cfg.get("attn_window"):
        missing.append("sliding-window attention")
    # Even a plain rope/nope softmax model uses the research surround (GQA + output gate) that the
    # legacy engine may not match exactly, so we always treat atma research ckpts as unsupported.
    if cfg and not missing:
        missing.append("research attention surround (GQA + output gate) — verify against eval.py first")
    return missing


class EvalModel:
    """Autoregressive generation adapter over inference.LLM for the benchmark harness.

    Usage (once inference is ported):
        m = EvalModel("checkpoints/<run_id>", max_tokens=16)
        texts = m.generate(["...prompt..."])          # list[str]

    Until then, construction prints the WIP disclaimer; pass strict=True to hard-fail instead
    of producing placeholder numbers.
    """

    def __init__(self, model_path: str, *, temperature: float = 0.0, max_tokens: int = 16,
                 strict: bool = False, quiet: bool = False, **llm_kwargs):
        self.model_path = model_path
        self.cfg = read_checkpoint_config(model_path)
        self.temperature = temperature           # 0.0 => greedy (engine must treat <=0 as argmax)
        self.max_tokens = max_tokens
        self._llm = None
        self._llm_kwargs = llm_kwargs
        self.wip = unsupported_features(self.cfg)
        if not quiet:
            self._announce(strict)
        if strict and self.wip:
            raise NotImplementedError(
                "Inference engine does not implement: " + "; ".join(self.wip)
                + " — see docs/inference.md. Re-run without --strict to produce placeholder numbers.")

    def _announce(self, strict):
        print(_BANNER)
        print("benchmarks.EvalModel - INFERENCE PORT WIP")
        if self.wip:
            print("The paged inference engine does NOT yet run, for this checkpoint:")
            for f in self.wip:
                print(f"  - {f}")
            print("=> Generations are INVALID and all scores are PLACEHOLDERS until the inference")
            print("   port is finished (docs/inference.md). " + ("Aborting (strict)." if strict
                  else "Proceeding with placeholder numbers."))
        else:
            print("No checkpoint config found / no unsupported features flagged — proceeding.")
        print(_BANNER)

    def load(self):
        """Construct the underlying inference.LLM lazily (needs CUDA + transformers + a ckpt)."""
        if self._llm is not None:
            return self
        from inference import LLM, SamplingParams  # noqa: F401  (import-time CUDA/transformers)
        if LLM is None:
            raise RuntimeError("inference.LLM is unavailable (transformers not importable).")
        self._llm = LLM(self.model_path, **self._llm_kwargs)
        return self

    def generate(self, prompts, max_tokens=None, temperature=None, use_tqdm=False):
        """list[str] prompts -> list[str] generated continuations (greedy by default)."""
        self.load()
        from inference import SamplingParams
        sp = SamplingParams(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        outs = self._llm.generate(list(prompts), sp, use_tqdm=use_tqdm)
        return [o["text"] for o in outs]
