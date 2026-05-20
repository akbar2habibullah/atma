import threading

class Context:
    def __init__(
        self,
        is_prefill: bool,
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping = None,
        context_lens = None,
        block_tables = None,
    ):
        self.is_prefill = is_prefill
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables


_local = threading.local()


def set_context(
    is_prefill: bool,
    cu_seqlens_q = None,
    cu_seqlens_k = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping = None,
    context_lens = None,
    block_tables = None,
):
    _local.context = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def get_context() -> Context:
    ctx = getattr(_local, "context", None)
    if ctx is None:
        raise RuntimeError("No context has been set. Ensure set_context is called before running the model layers.")
    return ctx


def reset_context():
    if hasattr(_local, "context"):
        del _local.context
