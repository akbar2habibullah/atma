"""Benchmark-only serving facade using the production scheduler contract."""

import atexit
from dataclasses import fields
from time import perf_counter
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from inference.config import Config
from inference.engine.sequence import Sequence
from inference.engine.scheduler import Scheduler
from inference.engine.block_manager import BlockManager
from inference.sampling_params import SamplingParams
from baseline_inference.runner import make_runner


class NoPrefixBlockManager(BlockManager):
    """Stateful baselines cannot safely reuse cache blocks without recurrent states."""

    def can_allocate(self, seq):
        return 0 if len(self.free_block_ids) >= seq.num_blocks else -1


class BaselineLLM:
    def __init__(self, model, **kwargs):
        names = {f.name for f in fields(Config)}
        kw = {k: v for k, v in kwargs.items() if k in names}
        self.config = Config(model, **kw)
        Sequence.block_size = self.config.kvcache_block_size
        self.model_runner = make_runner(self.config)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        self.scheduler.block_manager = NoPrefixBlockManager(
            self.config.num_kvcache_blocks, self.config.kvcache_block_size
        )
        self._last_metrics = None
        atexit.register(self.exit)

    @property
    def last_metrics(self):
        return self._last_metrics

    def exit(self):
        if getattr(self, "model_runner", None) is not None:
            self.model_runner.exit()
            self.model_runner = None

    def add_request(self, prompt, sp):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sp))

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        n = sum(s.num_scheduled_tokens for s in seqs) if prefill else -len(seqs)
        ids = self.model_runner.run(seqs, prefill)
        self.scheduler.postprocess(seqs, ids, prefill)
        for s in seqs:
            if s.is_finished:
                self.model_runner.free_seq_slot(s)
        return [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished], n

    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        sampling_params = sampling_params or SamplingParams()
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )
        for p, sp in zip(prompts, params):
            self.add_request(p, sp)
        bar = tqdm(total=len(prompts), disable=not use_tqdm)
        outs = {}
        pt = dt = 0.0
        pn = dn = 0
        while not self.scheduler.is_finished():
            if torch.cuda.is_available():
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                done, n = self.step()
                b.record()
                b.synchronize()
                elapsed = a.elapsed_time(b) / 1000
            else:
                t = perf_counter()
                done, n = self.step()
                elapsed = perf_counter() - t
            if n > 0:
                pn += n
                pt += elapsed
            elif n < 0:
                dn -= n
                dt += elapsed
            for sid, toks in done:
                outs[sid] = toks
                bar.update(1)
        bar.close()
        self._last_metrics = {
            "prefill_throughput": pn / pt if pt else 0.0,
            "decode_throughput": dn / dt if dt else 0.0,
            "prefill_tokens": pn,
            "decode_tokens": dn,
            "prefill_time": pt,
            "decode_time": dt,
        }
        return [
            {"text": self.tokenizer.decode(outs[i]), "token_ids": outs[i]}
            for i in sorted(outs)
        ]
