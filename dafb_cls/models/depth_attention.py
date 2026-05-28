import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class DepthAttentionBlock(nn.Module):
    def __init__(self, dim: int, zero_init: bool = True):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(dim) if zero_init else torch.randn(dim) * 0.02)

    def forward(self, block_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = block_features.shape
        keys = self.norm(block_features)
        scores = torch.einsum("d,bld->bl", self.query, keys)
        weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bl,bld->bd", weights, block_features)
        return output, weights


class ForegroundBackgroundDepthAttention(nn.Module):
    def __init__(self, dim: int, zero_init: bool = True, share: bool = False):
        super().__init__()
        self.dim = dim
        self.share = share
        self.fg_attn = DepthAttentionBlock(dim, zero_init=zero_init)
        if share:
            self.bg_attn = self.fg_attn
        else:
            self.bg_attn = DepthAttentionBlock(dim, zero_init=zero_init)

    def forward(
        self,
        B_F: torch.Tensor,
        B_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C_F, beta_F = self.fg_attn(B_F)
        C_B, beta_B = self.bg_attn(B_B)
        return C_F, C_B, beta_F, beta_B
