from collections import deque

import torch

from inference.config import Config
from inference.engine.sequence import Sequence
from inference.models.atma import Atma
from inference.layers.sampler import Sampler
from inference.utils.context import set_context, get_context, reset_context
from inference.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int = 0, event=None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModelRunner rank {rank} starting on device: {self.device}")

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        if self.device.type == "cuda":
            torch.set_default_device("cuda")

        self.model = Atma(hf_config)

        try:
            load_model(self.model, config.model, strict=config.strict_weights)
        except Exception as e:
            if config.strict_weights:
                raise RuntimeError(f"strict checkpoint loading failed: {e}") from e
            print(f"Weight loading failed ({e}). Using random weights.")

        self.sampler = Sampler()
        self.model.eval()

        # State tables first: the fp32 Titans memory tables can be GB-scale, and
        # allocate_kv_cache sizes the KV cache from the memory left after them.
        self.allocate_conv_state_tables()
        self.allocate_kv_cache()
        self.warmup_model()
        if not self.enforce_eager and self.device.type == "cuda":
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        if not self.enforce_eager and self.device.type == "cuda":
            if hasattr(self, "graphs"):
                del self.graphs
            if hasattr(self, "graph_pool"):
                del self.graph_pool
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"ModelRunner has no method '{method_name}'")
        return method(*args)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        self.attn_modules = [
            m for m in self.model.modules()
            if hasattr(m, "k_cache") and hasattr(m, "v_cache")
        ]
        num_attn_layers = len(self.attn_modules)
        if num_attn_layers == 0:
            print("Warning: no attention layers found — KV cache not allocated.")
            return

        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim
        itemsize = hf_config.dtype.itemsize
        block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * itemsize

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # warmup pass to measure peak model memory
            seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
            num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
            dummy_seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
            for s in dummy_seqs:
                s.num_scheduled_tokens = seq_len
                s.seq_slot = 0
            # Need a dummy conv_state_tables for warmup
            dummy_cst = {
                f"conv_{i}_gated": torch.zeros(
                    1, hf_config.hidden_size, hf_config.conv_kernel_size - 1, device="cuda"
                )
                for i in range(hf_config.num_hidden_layers) if i % 4 != 2
            }
            kv_hidden = hf_config.num_key_value_heads * hf_config.head_dim
            for i in range(hf_config.num_hidden_layers):
                if i % 4 == 2:
                    for s in ("q", "k", "v"):
                        dim = hf_config.hidden_size if s == "q" else kv_hidden
                        dummy_cst[f"attn_{i}_{s}"] = torch.zeros(
                            1, dim, hf_config.attn_kernel_size - 1, device="cuda"
                        )
                    if hf_config.mem_enabled:
                        dummy_cst[f"mem_{i}"] = torch.zeros(
                            1, hf_config.num_attention_heads, hf_config.head_dim,
                            hf_config.head_dim, dtype=torch.float32, device="cuda"
                        )
            seqlens_q = [seq_len] * num_seqs
            set_context(True,
                cu_seqlens_q=torch.tensor([i * seq_len for i in range(num_seqs + 1)], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([i * seq_len for i in range(num_seqs + 1)], dtype=torch.int32),
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                slot_mapping=torch.zeros(num_seqs * seq_len, dtype=torch.int32),
                seqlens_q=seqlens_q,
                conv_state_tables=dummy_cst,
            )
            ctx = get_context()
            ctx.seqs = dummy_seqs
            with torch.inference_mode():
                self.model(torch.zeros(num_seqs * seq_len, dtype=torch.int64))
            reset_context()
            torch.cuda.empty_cache()

            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            available = int(total * config.gpu_memory_utilization - used - peak + current)
            config.num_kvcache_blocks = max(16, available // block_bytes)
        else:
            config.num_kvcache_blocks = 128

        print(f"KV cache: {num_attn_layers} layers, {config.num_kvcache_blocks} blocks × {self.block_size} "
              f"({block_bytes * config.num_kvcache_blocks / 1e6:.1f} MB)")

        self.kv_cache = torch.zeros(
            2, num_attn_layers, config.num_kvcache_blocks, self.block_size,
            num_kv_heads, head_dim, dtype=hf_config.dtype,
        )
        for idx, module in enumerate(self.attn_modules):
            module.k_cache = self.kv_cache[0, idx]
            module.v_cache = self.kv_cache[1, idx]

    # ------------------------------------------------------------------
    # Conv state tables
    # ------------------------------------------------------------------

    def allocate_conv_state_tables(self):
        """Allocate centralized GPU per-sequence state tables.

        Conv states: (max_seqs, hdim, ks-1) per conv/canon key, model dtype.
        Titans memory states (mem_enabled): (max_seqs, H, dk, dv) per attention layer,
        fp32, in FLA's native [K, V] state layout (the torch fallback's (dv, dk) state is
        the transpose — see inference/models/atma.py). FLA keeps recurrent states fp32."""
        hf = self.config.hf_config
        hidden = hf.hidden_size
        attn_ks = hf.attn_kernel_size
        conv_ks = hf.conv_kernel_size
        max_seqs = self.config.max_num_seqs

        kv_hidden = hf.num_key_value_heads * hf.head_dim
        self.conv_state_tables: dict[str, torch.Tensor] = {}
        for i in range(hf.num_hidden_layers):
            if i % 4 == 2:  # attention layer
                for suffix in ("q", "k", "v"):
                    key = f"attn_{i}_{suffix}"
                    dim = hidden if suffix == "q" else kv_hidden
                    self.conv_state_tables[key] = torch.zeros(
                        max_seqs, dim, attn_ks - 1, dtype=hf.dtype,
                    )
                if hf.mem_enabled:
                    self.conv_state_tables[f"mem_{i}"] = torch.zeros(
                        max_seqs, hf.num_attention_heads, hf.head_dim, hf.head_dim,
                        dtype=torch.float32,
                    )
            else:            # LFM2 conv layer
                key = f"conv_{i}_gated"
                self.conv_state_tables[key] = torch.zeros(
                    max_seqs, hidden, conv_ks - 1, dtype=hf.dtype,
                )

        # Sequence slot free list
        self._free_slots: deque[int] = deque(range(max_seqs))
        print(f"Per-seq state tables (conv + mem): {len(self.conv_state_tables)} keys, "
              f"{sum(t.numel() * t.element_size() for t in self.conv_state_tables.values()) / 1e6:.1f} MB")

    def alloc_seq_slot(self, seq: Sequence) -> int:
        slot = self._free_slots.popleft()
        seq.seq_slot = slot
        return slot

    def free_seq_slot(self, seq: Sequence):
        if seq.seq_slot < 0:
            return
        for tbl in self.conv_state_tables.values():
            tbl[seq.seq_slot].zero_()
        self._free_slots.append(seq.seq_slot)
        seq.seq_slot = -1

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _make_warmup_seqs(self, seq_len: int):
        prefill_seq = Sequence([0] * seq_len)
        prefill_seq.num_scheduled_tokens = seq_len
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        prefill_seq.block_table = list(range(min(num_blocks, self.config.num_kvcache_blocks)))

        decode_seq = Sequence([0])
        decode_seq.num_scheduled_tokens = 1
        decode_seq.block_table = [0]

        return [prefill_seq], [decode_seq]

    def warmup_model(self):
        print("Warming up model...")
        seq_len = min(64, self.config.max_num_batched_tokens, self.config.max_model_len)
        prefill_seqs, decode_seqs = self._make_warmup_seqs(seq_len)
        try:
            with torch.inference_mode():
                self.run(prefill_seqs, is_prefill=True)
                # These warm up independent paths; release the prefill slot so a
                # max_num_seqs=1 engine can allocate the decode warm-up sequence.
                for seq in prefill_seqs:
                    if seq.seq_slot >= 0:
                        self.free_seq_slot(seq)
                self.run(decode_seqs, is_prefill=False)
        finally:
            for seq in prefill_seqs + decode_seqs:
                if seq.seq_slot >= 0:
                    self.free_seq_slot(seq)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("Warmup complete.")

    # ------------------------------------------------------------------
    # CUDA graph capture (decode only)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture_cudagraph(self):
        # Decode runs the paged polar Triton kernel (kernel/polar_triton.py); the
        # per-sequence CPU gather fallback is not graph-compatible.
        from inference.models.atma import polar_attention_decode, HAS_TRITON
        if polar_attention_decode is None or not HAS_TRITON:
            print("Skipping CUDA graph capture: polar Triton decode kernel unavailable.")
            return

        hf = self.config.hf_config
        max_bs = self.config.max_num_seqs
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

        input_ids    = torch.zeros(max_bs, dtype=torch.int64)
        positions    = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        seq_slots    = torch.zeros(max_bs, dtype=torch.int64)
        outputs      = torch.zeros(max_bs, hf.hidden_size)

        graph_candidates = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs]
        self.graph_bs   = sorted({bs for bs in graph_candidates if bs <= max_bs})
        self.graphs     = {}
        self.graph_pool = None

        print("Capturing CUDA graphs for decode...")
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                seq_slots=seq_slots[:bs],
                conv_state_tables=self.conv_state_tables,
            )
            # Warmup run (not captured)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            seq_slots=seq_slots,
            outputs=outputs,
        )
        print(f"CUDA graphs captured for batch sizes: {self.graph_bs}")

    # ------------------------------------------------------------------
    # Tensor preparation
    # ------------------------------------------------------------------

    def _to_cuda(self, t: torch.Tensor) -> torch.Tensor:
        if t.is_cuda:
            return t
        if self.device.type == "cuda":
            return t.pin_memory().cuda(non_blocking=True)
        return t.to(self.device)

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0
        rows = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._to_cuda(torch.tensor(rows, dtype=torch.int32))

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1] and seqs and seqs[0].block_table:
            block_tables = self.prepare_block_tables(seqs)

        seqlens_q = [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]

        # Dense v1 is deliberately narrow: fresh, complete, equal-length prompts.
        # A prefix-cache hit makes num_cached_tokens nonzero, while chunking makes
        # num_scheduled_tokens differ from num_tokens. Both use the packed fallback.
        dense_prefill = (
            len(seqs) > 1
            and seqlens_q[0] > 0
            and len(set(seqlens_q)) == 1
            and all(seq.num_cached_tokens == 0 for seq in seqs)
            and all(seq.num_scheduled_tokens == seq.num_tokens for seq in seqs)
            and sum(seqlens_q) <= self.config.max_num_batched_tokens
        )
        grouped_polar_prefill = (
            len(seqs) > 1
            and not dense_prefill
            and all(length > 0 for length in seqlens_q)
            and all(seq.num_cached_tokens == 0 for seq in seqs)
            and all(seq.num_scheduled_tokens == seq.num_tokens for seq in seqs)
            and sum(seqlens_q) <= self.config.max_num_batched_tokens
            and self.device.type == "cuda"
        )

        # L40S BF16 Polar uses 128 query rows per tile. Each entry maps one
        # independently scheduled tile to its packed sequence storage.
        tile_seq_starts, tile_q_starts, tile_seq_lens = [], [], []
        token_seq_starts, token_seq_ends, token_seq_slots = [], [], []
        if grouped_polar_prefill:
            packed_start = 0
            for seq, length in zip(seqs, seqlens_q):
                for query_start in range(0, length, 128):
                    tile_seq_starts.append(packed_start)
                    tile_q_starts.append(query_start)
                    tile_seq_lens.append(length)
                token_seq_starts.extend([packed_start] * length)
                token_seq_ends.extend([packed_start + length] * length)
                token_seq_slots.extend([seq.seq_slot] * length)
                packed_start += length

        input_ids_t    = self._to_cuda(torch.tensor(input_ids,    dtype=torch.int64))
        positions_t    = self._to_cuda(torch.tensor(positions,    dtype=torch.int64))
        cu_seqlens_q_t = self._to_cuda(torch.tensor(cu_seqlens_q, dtype=torch.int32))
        cu_seqlens_k_t = self._to_cuda(torch.tensor(cu_seqlens_k, dtype=torch.int32))
        slot_mapping_t = self._to_cuda(torch.tensor(slot_mapping, dtype=torch.int32))
        seq_slots_t = (self._to_cuda(torch.tensor([seq.seq_slot for seq in seqs], dtype=torch.int64))
                       if dense_prefill or grouped_polar_prefill else None)
        tile_seq_starts_t = (self._to_cuda(torch.tensor(tile_seq_starts, dtype=torch.int32))
                             if grouped_polar_prefill else None)
        tile_q_starts_t = (self._to_cuda(torch.tensor(tile_q_starts, dtype=torch.int32))
                           if grouped_polar_prefill else None)
        tile_seq_lens_t = (self._to_cuda(torch.tensor(tile_seq_lens, dtype=torch.int32))
                          if grouped_polar_prefill else None)
        token_seq_starts_t = (self._to_cuda(torch.tensor(token_seq_starts, dtype=torch.int32))
                              if grouped_polar_prefill else None)
        token_seq_ends_t = (self._to_cuda(torch.tensor(token_seq_ends, dtype=torch.int32))
                            if grouped_polar_prefill else None)
        token_seq_slots_t = (self._to_cuda(torch.tensor(token_seq_slots, dtype=torch.int32))
                             if grouped_polar_prefill else None)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_t,
            block_tables=block_tables,
            seqlens_q=seqlens_q,
            conv_state_tables=self.conv_state_tables,
            seq_slots=seq_slots_t,
            dense_prefill=dense_prefill,
            dense_batch_size=len(seqs) if dense_prefill else 0,
            dense_seq_len=seqlens_q[0] if dense_prefill else 0,
            grouped_polar_prefill=grouped_polar_prefill,
            polar_tile_seq_starts=tile_seq_starts_t,
            polar_tile_q_starts=tile_q_starts_t,
            polar_tile_seq_lens=tile_seq_lens_t,
            token_seq_starts=token_seq_starts_t,
            token_seq_ends=token_seq_ends_t,
            token_seq_slots=token_seq_slots_t,
        )
        get_context().seqs = seqs
        return input_ids_t, positions_t

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids    = []
        positions    = []
        slot_mapping = []
        context_lens = []
        slot_indices = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            slot_indices.append(seq.seq_slot)

        input_ids_t    = self._to_cuda(torch.tensor(input_ids,    dtype=torch.int64))
        positions_t    = self._to_cuda(torch.tensor(positions,    dtype=torch.int64))
        slot_mapping_t = self._to_cuda(torch.tensor(slot_mapping, dtype=torch.int32))
        context_lens_t = self._to_cuda(torch.tensor(context_lens, dtype=torch.int32))
        block_tables_t = self.prepare_block_tables(seqs)
        seq_slots_t    = self._to_cuda(torch.tensor(slot_indices, dtype=torch.int64))

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping_t,
            context_lens=context_lens_t,
            block_tables=block_tables_t,
            seq_slots=seq_slots_t,
            conv_state_tables=self.conv_state_tables,
        )
        return input_ids_t, positions_t

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return self._to_cuda(torch.tensor(temperatures, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Forward pass (uses CUDA graph for decode when available)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or not hasattr(self, "graphs") or input_ids.size(0) > self.config.max_num_seqs:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        gv = self.graph_vars

        gv["input_ids"][:bs]    = input_ids
        gv["positions"][:bs]    = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = context.context_lens
        n_blocks = context.block_tables.size(1)
        gv["block_tables"][:bs, :n_blocks] = context.block_tables
        if n_blocks < gv["block_tables"].size(1):
            gv["block_tables"][:bs, n_blocks:].zero_()
        gv["seq_slots"][:bs]    = context.seq_slots

        graph.replay()
        return self.model.compute_logits(gv["outputs"][:bs])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # Assign conv state slots to new sequences
        for seq in seqs:
            if seq.seq_slot < 0:
                self.alloc_seq_slot(seq)

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        reset_context()
        return token_ids
