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
    num_random_keys: int = 0
    attn_online: bool = False   # Polar attn: stream keys in blocks (O(T*k_block) memory)
    attn_k_block: int = 512
    # Polar attn core implementation:
    #   "torch"  -> materialized polar_reduce (or streamed polar_attention_online if attn_online)
    #   "triton" -> FlashAttention-style Triton kernel (kernel/polar_triton.py); CUDA only,
    #               falls back to the torch path on CPU / when triton is unavailable.
    attn_kernel: str = "triton"
    # MAG compression memory (Titans-style linear gated-delta) + sliding window.
    # Defaults leave the model byte-identical to plain polar attention (no window, no
    # memory). Enable both together for the MAG configuration.
    attn_window: int | None = None   # train-time causal sliding window for the polar core
    mem_enabled: bool = False        # add the Titans memory branch (out += mem)
    mem_chunk: int = 64              # chunk size for the gated-delta parallel scan
    mem_gamma_bias: float = 3.9      # retention logit init: sigmoid(3.9) ~ 0.98 (long horizon)
    mem_beta_bias: float = 0.0       # write-strength logit init: sigmoid(0) = 0.5

    @property
    def num_attention_heads(self) -> int:
        return self.hidden_size // self.head_dim

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads // 4  # GQA 1:4 ratio
