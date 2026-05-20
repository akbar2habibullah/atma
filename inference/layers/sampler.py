import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Prevent division by zero or extremely low temperature issues
        temps = temperatures.unsqueeze(dim=1).clamp_min(1e-5)
        scaled_logits = logits.float() / temps
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Gumbel-Max trick for sampling
        gumbel_noise = -torch.empty_like(probs).exponential_(1).log().clamp_min(1e-10)
        sample_tokens = (probs / gumbel_noise).argmax(dim=-1)
        return sample_tokens
