from dataclasses import dataclass, field
import torch

@dataclass
class AtmaHFConfig:
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    head_dim: int = 128
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False

@dataclass
class Config:
    model: str = "gpt2"
    tensor_parallel_size: int = 1
    kvcache_block_size: int = 16
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_model_len: int = 1024
    enforce_eager: bool = False
    eos: int = 50256
    num_kvcache_blocks: int = 0
    hf_config: AtmaHFConfig = field(default_factory=AtmaHFConfig)
