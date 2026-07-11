# `kernel/cross_entropy` - soft-capped linear cross entropy

This package contains the ATMA loss-head variant of Cut Cross-Entropy. It avoids
storing the full `tokens x vocab` logits tensor and folds the model's logits
softcap directly into the loss:

```python
from kernel.cross_entropy import softcap_linear_cross_entropy

loss = softcap_linear_cross_entropy(
    hidden_states,             # (..., hidden)
    model.proj.weight,         # (vocab, hidden)
    targets,                   # (...)
    model.proj.bias,
    softcap=15.0,
    reduction="sum",
)
```

Current implementation:

- CUDA/Triton forward streams over vocabulary blocks and writes only per-token
  losses plus log-sum-exp values.
- Backward recomputes logits in bounded token/vocab chunks, accumulating exact
  gradients for hidden states, classifier weights, and classifier bias.
- CPU and non-Triton CUDA fallback use the same chunked math.
- `impl="eager"` materializes logits and is kept as a parity reference.

For the default training microbatch (`mbs=8`, `seq_len=1024`, `vocab=50304`), a
single fp32 logits-sized tensor is about 1.54 GiB. This path replaces that with
`token_chunk_size * vocab_chunk_size` temporary logits during backward and no
full logits tensor in forward.
