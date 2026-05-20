from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False
