import threading


class Context:
    def __init__(
        self,
        is_prefill: bool,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping=None,
        context_lens=None,
        block_tables=None,
        seqlens_q: list = None,
        context_lens_list: list = None,
        conv_state_tables: dict = None,
        seq_slots=None,
        dense_prefill: bool = False,
        dense_batch_size: int = 0,
        dense_seq_len: int = 0,
    ):
        self.is_prefill = is_prefill
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.seqlens_q = seqlens_q or []
        self.context_lens_list = context_lens_list or []
        # Centralized GPU conv state tables — dict[str, Tensor(max_seqs, hdim, ks-1)]
        self.conv_state_tables = conv_state_tables
        # GPU tensor (bs,) of sequence slot indices for CUDA-graph-compatible conv state I/O
        self.seq_slots = seq_slots
        # Fresh, equal-length prefills stay flat at the model boundary while
        # layers view their storage as [B, T, ...] and launch batched kernels.
        self.dense_prefill = dense_prefill
        self.dense_batch_size = dense_batch_size
        self.dense_seq_len = dense_seq_len


_local = threading.local()


def set_context(
    is_prefill: bool,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    seqlens_q: list = None,
    context_lens_list: list = None,
    conv_state_tables: dict = None,
    seq_slots=None,
    dense_prefill: bool = False,
    dense_batch_size: int = 0,
    dense_seq_len: int = 0,
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
        seqlens_q=seqlens_q,
        context_lens_list=context_lens_list,
        conv_state_tables=conv_state_tables,
        seq_slots=seq_slots,
        dense_prefill=dense_prefill,
        dense_batch_size=dense_batch_size,
        dense_seq_len=dense_seq_len,
    )


def get_context() -> Context:
    ctx = getattr(_local, "context", None)
    if ctx is None:
        raise RuntimeError("No context has been set. Call set_context before running model layers.")
    return ctx


def reset_context():
    if hasattr(_local, "context"):
        del _local.context
