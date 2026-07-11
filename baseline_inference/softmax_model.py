"""Checkpoint-compatible NoPE/RoPE serving model, isolated from inference.models."""

import torch
from torch import nn
import torch.nn.functional as F

from inference.layers.attention import Attention, store_kvcache
from inference.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from inference.models.atma import (
    AtmaAttention,
    AtmaLFM2Conv,
    InferenceMLP,
    _gpu_conv_step,
    prefill_causal_conv1d,
    prefill_causal_conv1d_dense,
)
from inference.layers.linear import ReplicatedLinear
from inference.utils.context import get_context
from kernel.inference_ops_triton import softcap_logits
from model.blocks import AtmaAttnBase
from model.layers import RMSNorm
from baseline_inference.softmax_triton import paged_softmax_decode, HAS_TRITON


def _linear(i, o):
    return ReplicatedLinear(i, o, bias=True)


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq", torch.cat((freq, freq.new_zeros(dim // 4)))
        )

    def forward(self, x, positions):
        theta = positions.float()[:, None] * self.angular_freq[None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x.float().chunk(2, -1)
        y1 = x1 * cos[:, None] + x2 * sin[:, None]
        y2 = x1 * (-sin[:, None]) + x2 * cos[:, None]
        return torch.cat((y1, y2), -1).to(x.dtype)


class SoftmaxAttention(AtmaAttention):
    """Softmax mixer reusing only the validated conv/Titans state helpers."""

    def __init__(self, layer_idx, cfg):
        super().__init__(
            layer_idx,
            cfg.hidden_size,
            cfg.head_dim,
            cfg.num_key_value_heads,
            cfg.attn_kernel_size,
            cfg.attn_window,
            cfg.mem_enabled,
            cfg.mem_chunk,
            cfg.mem_gamma_bias,
            cfg.mem_beta_bias,
            cfg.mem_kernel,
        )
        # Remove polar-only checkpoint parameters.
        del self.mu_proj, self.v_null, self.null_base, self.null_slope_raw
        del self.len_gain_raw, self.mag_beta_raw
        self.pos = cfg.attn_type
        self.sdpa_scale = 0.12 if self.pos == "rope" else self.head_dim**-0.5
        self.rotary = Rotary(self.head_dim) if self.pos == "rope" else None
        if self.pos == "rope":
            self.canon_q = self.canon_k = self.canon_v = None

    def _rotate(self, x, positions):
        shape = x.shape
        return self.rotary(
            x.reshape(-1, shape[-2], shape[-1]), positions.reshape(-1)
        ).reshape(shape)

    def _attend(self, q, k, v, *, causal, q_start=0):
        # q/k/v [B,H,T,D], already GQA-expanded.
        Tq, Tk = q.shape[2], k.shape[2]
        mask = None
        if not causal or self.window is not None:
            qi = torch.arange(q_start, q_start + Tq, device=q.device)[:, None]
            ki = torch.arange(Tk, device=q.device)[None, :]
            valid = ki <= qi
            if self.window is not None:
                valid &= ki > qi - self.window
            mask = valid
            causal = False
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, scale=self.sdpa_scale
        )

    def forward(self, x):
        ctx = get_context()
        total = x.shape[0]
        H, D, G = self.num_heads, self.head_dim, self.num_heads // self.num_kv_heads
        qg = self.q(x).view(total, H, 2 * D)
        q, gate = qg.chunk(2, -1)
        k = self.k(x).view(total, self.num_kv_heads, D)
        v = self.v(x).view(total, self.num_kv_heads, D)
        q = F.rms_norm(q, (D,))
        k = F.rms_norm(k, (D,))
        table = (
            ctx.conv_state_tables.get(f"mem_{self.layer_idx}")
            if self.mem is not None
            else None
        )

        if ctx.is_prefill:
            if ctx.dense_prefill:
                B, T, slots = ctx.dense_batch_size, ctx.dense_seq_len, ctx.seq_slots
                qr, kr, vr = (
                    q.reshape(B, T, -1),
                    k.reshape(B, T, -1),
                    v.reshape(B, T, -1),
                )
                if self.pos == "nope":
                    wq, wk, wv = (
                        self.canon_q.weight.squeeze(1),
                        self.canon_k.weight.squeeze(1),
                        self.canon_v.weight.squeeze(1),
                    )
                    qr = qr + prefill_causal_conv1d_dense(
                        f"attn_{self.layer_idx}_q",
                        slots,
                        qr,
                        wq,
                        None,
                        ctx.conv_state_tables,
                    )
                    kr = kr + prefill_causal_conv1d_dense(
                        f"attn_{self.layer_idx}_k",
                        slots,
                        kr,
                        wk,
                        None,
                        ctx.conv_state_tables,
                    )
                    vr = vr + prefill_causal_conv1d_dense(
                        f"attn_{self.layer_idx}_v",
                        slots,
                        vr,
                        wv,
                        None,
                        ctx.conv_state_tables,
                    )
                qmem = qr.view(B, T, H, D).transpose(1, 2).contiguous()
                kmem = (
                    kr.view(B, T, self.num_kv_heads, D)
                    .repeat_interleave(G, 2)
                    .transpose(1, 2)
                    .contiguous()
                )
                vmem = (
                    vr.view(B, T, self.num_kv_heads, D)
                    .repeat_interleave(G, 2)
                    .transpose(1, 2)
                    .contiguous()
                )
                if self.pos == "rope":
                    pos = torch.arange(T, device=x.device).expand(B, T)
                    qa = (
                        self._rotate(q.view(B, T, H, D), pos)
                        .transpose(1, 2)
                        .contiguous()
                    )
                    ka0 = self._rotate(k.view(B, T, self.num_kv_heads, D), pos)
                else:
                    qa, ka0 = qmem, kr.view(B, T, self.num_kv_heads, D)
                va0 = vr.view(B, T, self.num_kv_heads, D)
                if self.attn.k_cache.numel():
                    store_kvcache(
                        ka0.reshape(total, self.num_kv_heads, D),
                        va0.reshape(total, self.num_kv_heads, D),
                        self.attn.k_cache,
                        self.attn.v_cache,
                        ctx.slot_mapping,
                    )
                ka = ka0.repeat_interleave(G, 2).transpose(1, 2).contiguous()
                va = va0.repeat_interleave(G, 2).transpose(1, 2).contiguous()
                y = (
                    self._attend(qa, ka, va, causal=True)
                    .transpose(1, 2)
                    .reshape(total, H * D)
                )
                out = self.proj(y * torch.sigmoid(gate.reshape(total, -1)))
                if self.mem is not None:
                    out += self._mem_prefill_dense(
                        x.view(B, T, -1), qmem, kmem, vmem, slots, table
                    )
                return out

            ys = []
            mems = []
            start = 0
            for i, T in enumerate(ctx.seqlens_q):
                seq = ctx.seqs[i]
                qs = q[start : start + T]
                ks = k[start : start + T]
                vs = v[start : start + T]
                if self.pos == "nope":
                    qs = qs.reshape(T, -1) + prefill_causal_conv1d(
                        f"attn_{self.layer_idx}_q",
                        seq,
                        qs.reshape(T, -1),
                        self.canon_q.weight.squeeze(1),
                        None,
                        ctx.conv_state_tables,
                    )
                    ks = ks.reshape(T, -1) + prefill_causal_conv1d(
                        f"attn_{self.layer_idx}_k",
                        seq,
                        ks.reshape(T, -1),
                        self.canon_k.weight.squeeze(1),
                        None,
                        ctx.conv_state_tables,
                    )
                    vs = vs.reshape(T, -1) + prefill_causal_conv1d(
                        f"attn_{self.layer_idx}_v",
                        seq,
                        vs.reshape(T, -1),
                        self.canon_v.weight.squeeze(1),
                        None,
                        ctx.conv_state_tables,
                    )
                    qs = qs.view(T, H, D)
                    ks = ks.view(T, self.num_kv_heads, D)
                    vs = vs.view(T, self.num_kv_heads, D)
                qmem = qs.transpose(0, 1).unsqueeze(0).contiguous()
                kmem = (
                    ks.repeat_interleave(G, 1).transpose(0, 1).unsqueeze(0).contiguous()
                )
                vmem = (
                    vs.repeat_interleave(G, 1).transpose(0, 1).unsqueeze(0).contiguous()
                )
                cached = seq.num_cached_tokens
                if self.pos == "rope":
                    pos = torch.arange(cached, cached + T, device=x.device)
                    qa = self.rotary(qs, pos)
                    ka = self.rotary(ks, pos)
                else:
                    qa, ka = qs, ks
                if self.attn.k_cache.numel():
                    store_kvcache(
                        ka,
                        vs,
                        self.attn.k_cache,
                        self.attn.v_cache,
                        ctx.slot_mapping[start : start + T],
                    )
                if cached:
                    bt = torch.as_tensor(seq.block_table, device=x.device)
                    bs = self.attn.k_cache.shape[1]
                    nb = (cached + bs - 1) // bs
                    kp = self.attn.k_cache[bt[:nb]].reshape(-1, self.num_kv_heads, D)[
                        :cached
                    ]
                    vp = self.attn.v_cache[bt[:nb]].reshape(-1, self.num_kv_heads, D)[
                        :cached
                    ]
                    ka = torch.cat((kp, ka))
                    vs2 = torch.cat((vp, vs))
                else:
                    vs2 = vs
                yh = self._attend(
                    qa.transpose(0, 1).unsqueeze(0),
                    ka.repeat_interleave(G, 1).transpose(0, 1).unsqueeze(0),
                    vs2.repeat_interleave(G, 1).transpose(0, 1).unsqueeze(0),
                    causal=cached == 0,
                    q_start=cached,
                )
                ys.append(yh.transpose(1, 2).reshape(T, H * D))
                if self.mem is not None:
                    mems.append(
                        self._mem_prefill(
                            seq, x[start : start + T], qmem, kmem, vmem, table
                        )
                    )
                start += T
            out = self.proj(torch.cat(ys) * torch.sigmoid(gate.reshape(total, -1)))
            if mems:
                out += torch.cat(mems)
            return out

        B = total
        slots = ctx.seq_slots
        if self.pos == "nope":
            qf = q.reshape(B, -1)
            kf = k.reshape(B, -1)
            vf = v.reshape(B, -1)
            qf += _gpu_conv_step(
                f"attn_{self.layer_idx}_q",
                slots,
                ctx.conv_state_tables,
                qf,
                self.canon_q.weight.squeeze(1),
            )
            kf += _gpu_conv_step(
                f"attn_{self.layer_idx}_k",
                slots,
                ctx.conv_state_tables,
                kf,
                self.canon_k.weight.squeeze(1),
            )
            vf += _gpu_conv_step(
                f"attn_{self.layer_idx}_v",
                slots,
                ctx.conv_state_tables,
                vf,
                self.canon_v.weight.squeeze(1),
            )
            qmem = qf.view(B, H, D)
            km0 = kf.view(B, self.num_kv_heads, D)
            vm0 = vf.view(B, self.num_kv_heads, D)
            qa, ka = qmem, km0
        else:
            qmem = q
            km0 = k
            vm0 = v
            pos = ctx.context_lens - 1
            qa = self.rotary(q, pos)
            ka = self.rotary(k, pos)
        store_kvcache(ka, vm0, self.attn.k_cache, self.attn.v_cache, ctx.slot_mapping)
        if HAS_TRITON and qa.is_cuda:
            y = paged_softmax_decode(
                qa,
                self.attn.k_cache,
                self.attn.v_cache,
                ctx.block_tables,
                ctx.context_lens,
                scale=self.sdpa_scale,
                window=self.window,
            )
        else:
            parts = []
            bs = self.attn.k_cache.shape[1]
            for b in range(B):
                n = int(ctx.context_lens[b])
                nb = (n + bs - 1) // bs
                kk = self.attn.k_cache[ctx.block_tables[b, :nb].long()].reshape(
                    -1, self.num_kv_heads, D
                )[:n]
                vv = self.attn.v_cache[ctx.block_tables[b, :nb].long()].reshape(
                    -1, self.num_kv_heads, D
                )[:n]
                if self.window is not None:
                    kk = kk[-self.window :]
                    vv = vv[-self.window :]
                parts.append(
                    F.scaled_dot_product_attention(
                        qa[b].unsqueeze(1),
                        kk.repeat_interleave(G, 1).transpose(0, 1),
                        vv.repeat_interleave(G, 1).transpose(0, 1),
                        scale=self.sdpa_scale,
                    ).squeeze(1)
                )
            y = torch.stack(parts)
        out = self.proj(y.reshape(B, -1) * torch.sigmoid(gate.reshape(B, -1)))
        if self.mem is not None:
            out += self._mem_decode(
                x,
                qmem,
                km0.repeat_interleave(G, 1),
                vm0.repeat_interleave(G, 1),
                slots,
                table,
            )
        return out


class Block(nn.Module):
    def __init__(self, idx, cfg):
        super().__init__()
        self.attn = (
            SoftmaxAttention(idx, cfg)
            if idx % 4 == 2
            else AtmaLFM2Conv(idx, cfg.hidden_size, cfg.conv_kernel_size)
        )
        self.mlp = InferenceMLP(cfg.hidden_size, linear_cls=_linear)
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SoftmaxLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [Block(i, cfg) for i in range(cfg.num_hidden_layers)]
        )
        self.proj = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, bias=True)
        self.norm = RMSNorm(cfg.hidden_size)

    def forward(self, input_ids, positions=None):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def compute_logits(self, x):
        logits = self.proj(x)
        return (
            softcap_logits(logits)
            if x.is_cuda
            else 15 * logits * (logits.square() + 225).rsqrt()
        )
