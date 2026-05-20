import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x = x + residual
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.type_as(x), eps=self.eps)


class MLP(nn.Module):

    def __init__(self, dim: int, linear_cls):
        super().__init__()
        hdim = 4 * dim
        self.fc = linear_cls(dim, hdim * 2)
        self.proj = linear_cls(hdim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.fc(x)
        x, gate = torch.chunk(x_gate, 2, dim=-1)
        x = gate * x.relu().square()
        return self.proj(x)
