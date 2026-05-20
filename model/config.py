from dataclasses import dataclass
import torch


@dataclass
class AtmaConfig:
    vocab_size: int = 50304
    num_hidden_layers: int = 16
    hidden_size: int = 1024
    head_dim: int = 128
    attn_kernel_size: int = 4
    conv_kernel_size: int = 3
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False

    @property
    def num_attention_heads(self) -> int:
        return self.hidden_size // self.head_dim

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads
