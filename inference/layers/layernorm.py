import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # High-performance PyTorch 2.4+ native rms_norm
            return F.rms_norm(x, (x.size(-1),), weight=self.weight.type_as(x), eps=self.eps)
        except AttributeError:
            # Pure PyTorch fallback for older versions
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            x_accumulated = x + residual
            return self.rms_forward(x_accumulated), x_accumulated
