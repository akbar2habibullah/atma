import torch
from torch import nn

from inference.config import Config
from inference.engine.sequence import Sequence
from inference.models.atma import Atma
from inference.layers.sampler import Sampler
from inference.utils.context import set_context, get_context, reset_context
from inference.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int = 0, event = None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        # Auto-detect hardware device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModelRunner rank {rank} starting on device: {self.device}")

        # Change default dtype safely
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        
        # Initialize inference optimized Atma model
        self.model = Atma(hf_config)
        self.model = self.model.to(self.device)
        
        # Load weights if config.model points to a valid file
        try:
            load_model(self.model, config.model)
        except Exception as e:
            print(f"No valid model weight path provided or failed loading: {e}. Running with randomly initialized weights.")
            
        self.sampler = Sampler().to(self.device)
        self.model.eval()

        self.allocate_kv_cache()
        self._try_compile_model()
        self.warmup_model()

        # Restore default dtype
        torch.set_default_dtype(default_dtype)

    def exit(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"ModelRunner has no method '{method_name}'")
        return method(*args)

    def _try_compile_model(self):
        """Wrap model with torch.compile; save original so warmup can fall back to eager."""
        if self.config.enforce_eager:
            print("enforce_eager=True, using eager mode.")
            return
        if not hasattr(torch, "compile"):
            print("torch.compile unavailable (requires PyTorch >= 2.0), using eager mode.")
            return
        try:
            self._orig_model = self.model
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                dynamic=True,
                fullgraph=False,
            )
            print("torch.compile enabled (mode=reduce-overhead, dynamic=True).")
        except Exception as e:
            print(f"torch.compile setup failed ({e}), using eager mode.")
            self.model = self._orig_model
            del self._orig_model

    def _make_warmup_seqs(self, seq_len: int):
        """Build minimal prefill and decode sequences for warmup passes."""
        prefill_seq = Sequence([0] * seq_len)
        prefill_seq.num_scheduled_tokens = seq_len
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        prefill_seq.block_table = list(range(min(num_blocks, self.config.num_kvcache_blocks)))

        decode_seq = Sequence([0])
        decode_seq.num_scheduled_tokens = 1
        decode_seq.block_table = [0]

        return [prefill_seq], [decode_seq]

    def warmup_model(self):
        """Warm up with prefill and decode passes; falls back to eager if torch.compile fails."""
        print("Warming up model...")
        seq_len = min(64, self.config.max_num_batched_tokens, self.config.max_model_len)

        prefill_seqs, decode_seqs = self._make_warmup_seqs(seq_len)
        try:
            with torch.inference_mode():
                self.run(prefill_seqs, is_prefill=True)
                self.run(decode_seqs, is_prefill=False)
            if hasattr(self, "_orig_model"):
                del self._orig_model
        except Exception as e:
            if hasattr(self, "_orig_model"):
                print(f"torch.compile warmup failed ({e}), falling back to eager mode.")
                self.model = self._orig_model
                del self._orig_model
                prefill_seqs, decode_seqs = self._make_warmup_seqs(seq_len)
                with torch.inference_mode():
                    self.run(prefill_seqs, is_prefill=True)
                    self.run(decode_seqs, is_prefill=False)
            else:
                raise

        # Warm up decode at typical batch sizes so torch.compile finishes
        # compilation before the first timed benchmark run.
        warmup_batch_sizes = [1, 4, 8, 16, 32]
        for bs in warmup_batch_sizes:
            _, decode_seqs = self._make_warmup_seqs(seq_len)
            # Replicate the single decode seq to reach the target batch size
            extra_seqs = []
            for j in range(bs - 1):
                _, extra = self._make_warmup_seqs(seq_len)
                extra_seqs.extend(extra)
            batch_decode_seqs = decode_seqs + extra_seqs
            try:
                with torch.inference_mode():
                    self.run(batch_decode_seqs, is_prefill=False)
            except Exception:
                pass  # skip if KV cache is too small for this batch

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("Warmup complete!")

    def allocate_kv_cache(self):
        """
        Allocates physical KV-cache page blocks.
        Optimized to only allocate blocks for blocks containing self-attention.
        """
        config = self.config
        hf_config = config.hf_config

        # Identify only the modules that actually possess KV-cache layers
        self.attn_modules = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                self.attn_modules.append(module)

        num_attn_layers = len(self.attn_modules)
        if num_attn_layers == 0:
            print("Warning: No self-attention layers found in the model!")
            return

        num_kv_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim
        itemsize = 2 if hf_config.dtype in [torch.bfloat16, torch.float16] else 4
        
        # Calculate bytes per block (2 matrices: Key and Value)
        block_bytes = (2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * itemsize)

        if self.device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                peak = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)
                current = torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
                available_bytes = int(total * config.gpu_memory_utilization - used - peak + current)
                config.num_kvcache_blocks = max(16, available_bytes // block_bytes)
            except Exception:
                config.num_kvcache_blocks = 256
        else:
            config.num_kvcache_blocks = 128

        print(f"Allocating KV cache for {num_attn_layers} attention layers: "
              f"{config.num_kvcache_blocks} blocks of size {self.block_size} (Block memory: {block_bytes * config.num_kvcache_blocks / 1e6:.2f}MB)")

        # kv_cache shape: (2, num_attn_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        self.kv_cache = torch.zeros(
            2,
            num_attn_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.dtype,
            device=self.device,
        )

        # Wire caches into each attention layer
        for idx, module in enumerate(self.attn_modules):
            module.k_cache = self.kv_cache[0, idx]
            module.v_cache = self.kv_cache[1, idx]

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=(self.device.type == "cuda"))

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

            # Map slots if sequence already has blocks allocated
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

        # Derive per-sequence lengths as Python ints before tensor conversion to
        # avoid Tensor.item() calls inside torch.compile (which cause graph breaks).
        seqlens_q = [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)

        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device)

        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        if block_tables is not None:
            block_tables = block_tables.to(self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            seqlens_q=seqlens_q,
        )
        get_context().seqs = seqs
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        context_lens_list = list(context_lens)  # Python ints before tensor conversion

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)

        block_tables = self.prepare_block_tables(seqs).to(self.device)

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            context_lens_list=context_lens_list,
        )
        get_context().seqs = seqs
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)

        # Run forward pass of Atma model
        logits = self.model(input_ids, positions)
        
        # Sample tokens
        token_ids = self.sampler(logits, temperatures).tolist()
        
        # Clean context for current run
        reset_context()
        
        return token_ids
