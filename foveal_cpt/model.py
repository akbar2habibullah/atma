from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kernel.cross_entropy import softcap_linear_cross_entropy
from train.model import Model

from .attention import FovealAttention
from .config import FovealConfig


def foveal_layers(model: Model) -> Iterable[FovealAttention]:
    for block in model.blocks:
        if isinstance(block.attn, FovealAttention):
            yield block.attn


class FovealCPTModel(nn.Module):
    """Memory-bounded training facade around the existing ATMA model body."""

    def __init__(self, base: Model, config: FovealConfig):
        super().__init__()
        self.base = base
        self.config = config

    def set_mode(self, mode: str) -> None:
        for layer in foveal_layers(self.base):
            layer.set_mode(mode)

    def set_step(self, step: int) -> None:
        for layer in foveal_layers(self.base):
            layer.set_step(step)

    def set_route(self, top_p: float, min_remote_pages: int, max_remote_pages: int) -> None:
        for layer in foveal_layers(self.base):
            layer.set_route(top_p, min_remote_pages, max_remote_pages)

    def index_parameters(self) -> list[nn.Parameter]:
        params = []
        for layer in foveal_layers(self.base):
            params.extend(layer.index_q.parameters())
            params.extend(layer.index_k.parameters())
            if layer.index_rotary is not None:
                params.extend(layer.index_rotary.parameters())
        return params

    def freeze_except_index(self) -> None:
        index = {id(param) for param in self.index_parameters()}
        for param in self.parameters():
            param.requires_grad_(id(param) in index)

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def _block(self, block: nn.Module, x: Tensor):
        if self.config.activation_checkpointing and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def features(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.base.embed(inputs)
        reg_loss = x.new_zeros((), dtype=torch.float32)
        index_loss = x.new_zeros((), dtype=torch.float32)
        for block in self.base.blocks:
            x, reg, align = self._block(block, x)
            reg_loss = reg_loss + reg.float()
            index_loss = index_loss + align.float()
        return self.base.norm(x), reg_loss / len(self.base.blocks), index_loss / self.base.num_attn_layers

    def forward(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden, reg_loss, index_loss = self.features(inputs)
        lm_loss = softcap_linear_cross_entropy(
            hidden,
            self.base.proj.weight,
            targets,
            self.base.proj.bias,
            reduction="sum",
            impl=self.config.xent_impl,
            token_chunk_size=self.config.xent_token_chunk,
            vocab_chunk_size=self.config.xent_vocab_chunk,
        )
        return lm_loss, reg_loss, index_loss

    def calibration_loss(self, inputs: Tensor) -> Tensor:
        _, _, index_loss = self.features(inputs)
        return index_loss

    def route_stats(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for layer in foveal_layers(self.base):
            for key, value in layer.last_stats.items():
                values.setdefault(key, []).append(float(value.float().mean().item()))
        return {key: sum(items) / len(items) for key, items in values.items() if items}
