from dataclasses import dataclass, field

from model.config import AtmaConfig


@dataclass
class Config:
    model: str = "gpt2"
    tensor_parallel_size: int = 1
    kvcache_block_size: int = 256
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384 * 8
    max_model_len: int = 4096
    enforce_eager: bool = False
    eos: int = 50256
    num_kvcache_blocks: int = -1
    hf_config: AtmaConfig = field(default_factory=AtmaConfig)
