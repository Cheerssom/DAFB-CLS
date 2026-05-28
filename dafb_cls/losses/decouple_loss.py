import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupleLoss(nn.Module):
    def __init__(self, method: str = "cosine_squared"):
        super().__init__()
        self.method = method

    def forward(self, C_F: torch.Tensor, C_B: torch.Tensor) -> torch.Tensor:
        if self.method == "cosine_squared":
            cos_sim = F.cosine_similarity(C_F, C_B, dim=-1)
            return (cos_sim ** 2).mean()
        elif self.method == "negative_cosine":
            cos_sim = F.cosine_similarity(C_F, C_B, dim=-1)
            return (1.0 + cos_sim).mean()
        elif self.method == "cross_covariance":
            C_F_centered = C_F - C_F.mean(dim=0, keepdim=True)
            C_B_centered = C_B - C_B.mean(dim=0, keepdim=True)
            cov = (C_F_centered * C_B_centered).mean(dim=0)
            return (cov ** 2).sum()
        else:
            raise ValueError(f"Unknown decouple method: {self.method}")
