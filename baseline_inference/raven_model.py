"""State-carrying Raven serving model for the isolated benchmark engine."""

import torch
from torch import nn
import torch.nn.functional as F
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa

from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.models.atma import AtmaAttention, AtmaLFM2Conv, InferenceMLP
from inference.utils.context import get_context
from kernel.inference_ops_triton import softcap_logits
from model.layers import RMSNorm
from raven_baseline.layers import RavenAttention, Linear


class RavenInferenceAttention(RavenAttention):
    _mem_use_fla = AtmaAttention._mem_use_fla
    _mem_prefill = AtmaAttention._mem_prefill
    _mem_prefill_dense = AtmaAttention._mem_prefill_dense
    _mem_decode = AtmaAttention._mem_decode

    def __init__(self, cfg, layer_idx, use_titans):
        super().__init__(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            num_slots=cfg["num_slots"],
            topk=cfg["topk"],
            feature_map=cfg["feature_map"],
            decay_type=cfg["decay_type"],
            router_score=cfg["router_score"],
            router_type=cfg["router_type"],
            add_gumbel_noise=cfg["add_gumbel_noise"],
            bias_rmm=cfg["bias_rmm"],
            gate_logit_normalizer=cfg["gate_logit_normalizer"],
            mem_enabled=use_titans,
            mem_chunk=cfg["mem_chunk"],
            mem_gamma_bias=cfg["mem_gamma_bias"],
            mem_beta_bias=cfg["mem_beta_bias"],
            mem_kernel=cfg["mem_kernel"],
        )
        self.layer_idx = layer_idx

    def _project(self, x):
        B, T, _ = x.shape
        H, D, G = self.num_heads, self.head_dim, self.num_kv_groups
        q = self.q_norm(self._feature_map(self.q_proj(x).view(B, T, H, D)))
        k = self.k_norm(
            self._feature_map(self.k_proj(x).view(B, T, self.num_kv_heads, D))
        )
        v = F.silu(self.v_proj(x).view(B, T, self.num_kv_heads, D))
        f, s = self._route(x)
        if G > 1:
            k = k.repeat_interleave(G, 2)
            v = v.repeat_interleave(G, 2)
        return q, k, v, f, s

    def _scan(self, x, slots, tables):
        q, k, v, f, s = self._project(x)
        initial = [
            tables[f"raven_k_{self.layer_idx}"][slots],
            tables[f"raven_v_{self.layer_idx}"][slots],
        ]
        fn = fused_recurrent_gsa if x.shape[1] <= 64 else chunk_gsa
        o, state = fn(
            q=q,
            k=k,
            v=v,
            s=s,
            g=f,
            scale=self.scale,
            initial_state=initial,
            output_final_state=True,
        )
        tables[f"raven_k_{self.layer_idx}"][slots] = state[0]
        tables[f"raven_v_{self.layer_idx}"][slots] = state[1]
        out = self.o_proj(
            self.o_norm(F.silu(o)).reshape(x.shape[0] * x.shape[1], self.hidden_size)
        )
        if self.mem is not None:
            if x.shape[0] > 1:
                out += self._mem_prefill_dense(
                    x,
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    slots,
                    tables[f"mem_{self.layer_idx}"],
                )
            else:
                out += self._mem_prefill(
                    get_context().seqs[0],
                    x[0],
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    tables[f"mem_{self.layer_idx}"],
                )
        return out

    def forward(self, x):
        ctx = get_context()
        tables = ctx.conv_state_tables
        if ctx.is_prefill:
            if ctx.dense_prefill:
                B, T = ctx.dense_batch_size, ctx.dense_seq_len
                return self._scan(x.view(B, T, -1), ctx.seq_slots, tables)
            parts = []
            start = 0
            for i, T in enumerate(ctx.seqlens_q):
                # Keep the matching sequence first for the inherited scalar Titans helper.
                old = ctx.seqs
                ctx.seqs = [old[i]]
                parts.append(
                    self._scan(
                        x[start : start + T].unsqueeze(0),
                        torch.tensor([old[i].seq_slot], device=x.device),
                        tables,
                    )
                )
                ctx.seqs = old
                start += T
            return torch.cat(parts)
        B = x.shape[0]
        q, k, v, f, s = self._project(x.unsqueeze(1))
        slots = ctx.seq_slots
        initial = [
            tables[f"raven_k_{self.layer_idx}"][slots],
            tables[f"raven_v_{self.layer_idx}"][slots],
        ]
        o, state = fused_recurrent_gsa(
            q=q,
            k=k,
            v=v,
            s=s,
            g=f,
            scale=self.scale,
            initial_state=initial,
            output_final_state=True,
        )
        tables[f"raven_k_{self.layer_idx}"][slots] = state[0]
        tables[f"raven_v_{self.layer_idx}"][slots] = state[1]
        out = self.o_proj(self.o_norm(F.silu(o)).reshape(B, self.hidden_size))
        if self.mem is not None:
            out += self._mem_decode(
                x, q[:, 0], k[:, 0], v[:, 0], slots, tables[f"mem_{self.layer_idx}"]
            )
        return out


class Block(nn.Module):
    def __init__(self, cfg, idx, use_conv, use_titans):
        super().__init__()
        self.attn = (
            AtmaLFM2Conv(idx, cfg["hidden_size"], cfg["conv_kernel_size"])
            if use_conv
            else RavenInferenceAttention(cfg, idx, use_titans)
        )
        self.mlp = InferenceMLP(cfg["hidden_size"], linear_cls=Linear)
        self.norm1 = RMSNorm(cfg["hidden_size"])
        self.norm2 = RMSNorm(cfg["hidden_size"])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RavenLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg["hidden_size"]
        self.embed = VocabParallelEmbedding(cfg["vocab_size"], D)
        arch = cfg["arch_type"]
        schedule = (
            [False] * cfg["num_hidden_layers"]
            if arch == "raven_native"
            else [i % 4 != 2 for i in range(cfg["num_hidden_layers"])]
        )
        self.blocks = nn.ModuleList(
            [
                Block(cfg, i, c, arch == "atma_raven_titans" and not c)
                for i, c in enumerate(schedule)
            ]
        )
        self.proj = ParallelLMHead(cfg["vocab_size"], D, bias=True)
        self.norm = RMSNorm(D)

    def forward(self, ids, positions=None):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def compute_logits(self, x):
        logits = self.proj(x)
        return (
            softcap_logits(logits)
            if logits.is_cuda
            else 15 * logits * (logits.square() + 225).rsqrt()
        )
