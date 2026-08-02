from collections import deque
from types import SimpleNamespace

import torch


def test_model_runner_warmup_supports_one_sequence_slot():
    from inference.engine.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        max_num_batched_tokens=64,
        max_model_len=64,
        num_kvcache_blocks=1,
    )
    runner.block_size = 256
    runner.device = torch.device("cpu")
    runner._free_slots = deque([0])
    runner.conv_state_tables = {}
    calls = []

    def run(seqs, is_prefill):
        calls.append(is_prefill)
        for seq in seqs:
            if seq.seq_slot < 0:
                runner.alloc_seq_slot(seq)

    runner.run = run
    runner.warmup_model()

    assert calls == [True, False]
    assert list(runner._free_slots) == [0]
