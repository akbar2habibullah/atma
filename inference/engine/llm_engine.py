import atexit
from dataclasses import fields
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from inference.config import Config
from inference.sampling_params import SamplingParams
from inference.engine.sequence import Sequence
from inference.engine.scheduler import Scheduler
from inference.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size
        
        self.model_runner = ModelRunner(self.config, 0)
        
        # Load fast tokenizer, falling back to 'gpt2' if local path fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        except Exception:
            print(f"Warning: Failed loading tokenizer from custom path '{self.config.model}'. "
                  f"Falling back to standard 'gpt2' tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams = None):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
            
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        
        # Call model runner to execute forward pass and return sampled token IDs
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # Release conv state slots for finished sequences
        for seq in seqs:
            if seq.is_finished and seq.seq_slot >= 0:
                self.model_runner.free_seq_slot(seq)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        total_prefill_time = 0.0
        total_decode_time = 0.0
        total_prefill_tokens = 0
        total_decode_tokens = 0
        prefill_throughput = decode_throughput = 0.

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
        
        while not self.is_finished():
            if use_cuda:
                _t0.record()
                output, num_tokens = self.step()
                _t1.record()
                _t1.synchronize()
                elapsed = _t0.elapsed_time(_t1) / 1000.0
            else:
                t = perf_counter()
                output, num_tokens = self.step()
                elapsed = perf_counter() - t
            if num_tokens > 0:
                total_prefill_time += elapsed
                total_prefill_tokens += num_tokens
                prefill_throughput = num_tokens / elapsed
            elif num_tokens < 0:
                total_decode_time += elapsed
                total_decode_tokens += -num_tokens
                decode_throughput = -num_tokens / elapsed
                
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
                
        pbar.close()

        self._last_metrics = {
            "prefill_throughput": total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0.,
            "decode_throughput": total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.,
            "prefill_tokens": total_prefill_tokens,
            "decode_tokens": total_decode_tokens,
            "prefill_time": total_prefill_time,
            "decode_time": total_decode_time,
        }
        
        sorted_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        decoded_outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in sorted_outputs
        ]
        return decoded_outputs
