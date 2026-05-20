from torch import nn


class AtmaConvBase(nn.Module):
    """Shared __init__ for LFM2 gated conv block. Subclass must implement forward()."""

    def __init__(self, dim: int, linear_cls, kernel_size: int = 3):
        super().__init__()
        self.hidden_size = dim
        self.kernel_size = kernel_size
        self.in_proj = linear_cls(dim, 3 * dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size - 1, groups=dim, bias=False)
        self.out_proj = linear_cls(dim, dim)

    def forward(self, x):
        raise NotImplementedError


class AtmaAttnBase(nn.Module):
    """Shared __init__ for Canon-B attention block. Subclass must implement forward()."""

    def __init__(self, dim: int, linear_cls, head_dim: int = 128, kernel_size: int = 4):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.hdim = self.num_heads * self.head_dim
        self.kernel_size = kernel_size

        self.q = linear_cls(dim, self.hdim * 2)
        self.k = linear_cls(dim, self.hdim)
        self.v = linear_cls(dim, self.hdim)
        self.canon_q = nn.Conv1d(self.hdim, self.hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=self.hdim, bias=False)
        self.canon_k = nn.Conv1d(self.hdim, self.hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=self.hdim, bias=False)
        self.canon_v = nn.Conv1d(self.hdim, self.hdim, kernel_size=kernel_size, padding=kernel_size - 1, groups=self.hdim, bias=False)
        self.proj = linear_cls(self.hdim, dim)

    def forward(self, x):
        raise NotImplementedError
