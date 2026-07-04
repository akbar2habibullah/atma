from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.layers import MLP, RMSNorm
from train.model import LFM2Conv
from train.reg import sigreg
from raven_baseline.layers import Linear, RavenAttention


LOGIT_SOFTCAP = 15.0


class RavenBlock(nn.Module):
    def __init__(self, cfg: dict, *, layer_idx: int, use_lfm2: bool, use_titans: bool):
        super().__init__()
        self.use_lfm2 = use_lfm2
        if use_lfm2:
            self.attn = LFM2Conv(cfg["hidden_size"], kernel_size=cfg["conv_kernel_size"])
        else:
            self.attn = RavenAttention(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg.get("num_kv_heads"),
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
        self.mlp = MLP(cfg["hidden_size"], linear_cls=Linear)
        self.norm1 = RMSNorm(cfg["hidden_size"])
        self.norm2 = RMSNorm(cfg["hidden_size"])
        self.reg_mode = cfg.get("reg_mode", "baseline")
        self.sketch_dim = cfg.get("sketch_dim", 64)
        self.layer_idx = layer_idx

    def forward(self, x: Tensor):
        y, align_loss = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        reg_loss = sigreg(x, self.reg_mode, self.sketch_dim)
        return x, reg_loss, align_loss


class RavenLM(nn.Module):
    """Atma-compatible LM wrapper for native Raven and Atma-Raven variants."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"]).bfloat16()
        arch = cfg["arch_type"]
        if arch == "raven_native":
            schedule = [False] * cfg["num_hidden_layers"]
        elif arch in {"atma_raven", "atma_raven_titans"}:
            schedule = [not (i % 4 == 2) for i in range(cfg["num_hidden_layers"])]
        else:
            raise ValueError(f"unsupported arch_type={arch}")
        use_titans = arch == "atma_raven_titans"
        self.blocks = nn.ModuleList([
            RavenBlock(cfg, layer_idx=i, use_lfm2=use_lfm2, use_titans=(use_titans and not use_lfm2))
            for i, use_lfm2 in enumerate(schedule)
        ])
        self.proj = Linear(cfg["hidden_size"], cfg["vocab_size"])
        self.norm = RMSNorm(cfg["hidden_size"])

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.embed(inputs)
        total_reg_loss = 0.0
        total_align_loss = 0.0
        n_mixer = 0
        for block in self.blocks:
            x, reg_loss, align_loss = block(x)
            total_reg_loss = total_reg_loss + reg_loss
            total_align_loss = total_align_loss + align_loss
            n_mixer += 1
        logits = self.proj(self.norm(x)).float()
        logits = LOGIT_SOFTCAP * logits * (logits.square() + LOGIT_SOFTCAP ** 2).rsqrt()
        loss = F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
        return loss, total_reg_loss / len(self.blocks), total_align_loss / max(n_mixer, 1)


def create_model(cfg: dict) -> RavenLM:
    return RavenLM(cfg)

