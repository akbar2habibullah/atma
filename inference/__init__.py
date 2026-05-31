from inference.sampling_params import SamplingParams

# The full vLLM-style engine (LLM) pulls in transformers (tokenizer). Import it
# lazily so the lightweight model/layer submodules (and inference.generate) remain
# usable without transformers installed.
try:
    from inference.llm import LLM
except Exception:  # transformers not available
    LLM = None
