import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class ForegroundnessHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_cues: int = 4,
        hidden_dim: int = 256,
        grid_size: int = 14,
        use_spatial_smoothing: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_cues = num_cues
        self.grid_size = grid_size
        self.use_spatial_smoothing = use_spatial_smoothing

        self.cue_weights = nn.Parameter(torch.ones(num_cues) / num_cues)

        self.mlp = nn.Sequential(
            nn.Linear(num_cues, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        if use_spatial_smoothing:
            self.spatial_smooth = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.spatial_smooth[0].weight, 1.0 / 9.0)

    def forward(
        self,
        frequency_cue: torch.Tensor,
        depth_cue: torch.Tensor,
        semantic_cue: torch.Tensor,
        spatial_cue: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N = frequency_cue.shape

        cues = torch.stack([frequency_cue, depth_cue, semantic_cue, spatial_cue], dim=-1)

        w = F.softmax(self.cue_weights, dim=0)
        weighted_cues = cues * w.unsqueeze(0).unsqueeze(0)
        F_score = weighted_cues.sum(dim=-1)

        cue_logits = self.mlp(cues).squeeze(-1)
        F_combined = F_score + cue_logits

        info = {
            "frequency_cue": frequency_cue,
            "depth_cue": depth_cue,
            "semantic_cue": semantic_cue,
            "spatial_cue": spatial_cue,
            "cue_weights": w.detach(),
            "raw_score": F_score,
            "mlp_score": cue_logits,
        }

        if self.use_spatial_smoothing:
            F_2d = F_combined.reshape(B, 1, self.grid_size, self.grid_size)
            F_smoothed = self.spatial_smooth(F_2d).reshape(B, N)
            F_combined = F_combined + 0.5 * F_smoothed
            info["smoothed_score"] = F_smoothed

        return F_combined, info
