import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TaskAdaptiveFusionHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.gate_predictor = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        C_F: torch.Tensor,
        C_B: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([C_F, C_B, global_feature], dim=-1)
        g = self.gate_predictor(gate_input)
        C = g * C_F + (1.0 - g) * C_B
        return C, g
