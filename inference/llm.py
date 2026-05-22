from inference.engine.llm_engine import LLMEngine
from inference.sampling_params import SamplingParams


class LLM:

    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(model, **kwargs)

    @property
    def last_metrics(self) -> dict | None:
        return getattr(self.engine, "_last_metrics", None)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Generates text completions for a list of prompts.
        Returns a list of dicts: [{"text": str, "token_ids": list[int]}]
        """
        return self.engine.generate(prompts, sampling_params, use_tqdm)
