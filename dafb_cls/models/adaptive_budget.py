import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveBudgetModule(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        temperature_init: float = 0.1,
        tau_from_global: bool = True,
        mask_type: str = "soft",
        topk_ratio: float = 0.3,
        use_score_stats: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.tau_from_global = tau_from_global
        self.mask_type = mask_type
        self.topk_ratio = topk_ratio
        self.use_score_stats = use_score_stats

        self.log_temperature = nn.Parameter(torch.tensor(float(temperature_init)).log())

        if tau_from_global:
            inp_dim = dim + 4 if use_score_stats else dim
            self.tau_predictor = nn.Sequential(
                nn.Linear(inp_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.tau = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        foreground_score: torch.Tensor,
        global_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        B, N = foreground_score.shape
        T = self.log_temperature.exp().clamp(min=0.01, max=1.0)

        if self.tau_from_global and global_feature is not None:
            if self.use_score_stats:
                stats = torch.stack([
                    foreground_score.mean(dim=1),
                    foreground_score.std(dim=1),
                    foreground_score.max(dim=1).values,
                    foreground_score.min(dim=1).values,
                ], dim=1)
                tau_input = torch.cat([global_feature, stats], dim=1)
            else:
                tau_input = global_feature
            tau = self.tau_predictor(tau_input).squeeze(-1).unsqueeze(1)
        else:
            tau = self.tau.expand(B).unsqueeze(1)

        fg_logits = (foreground_score - tau) / T

        if self.mask_type == "hard_topk":
            k = max(1, int(N * self.topk_ratio))
            _, topk_indices = foreground_score.topk(k, dim=1)
            mask_fg = torch.zeros_like(foreground_score)
            mask_fg.scatter_(1, topk_indices, 1.0)
        else:
            mask_fg = torch.sigmoid(fg_logits)

        info = {
            "tau": tau.squeeze(1).detach(),
            "temperature": T.detach(),
            "fg_ratio": mask_fg.mean(dim=1).detach(),
            "raw_mask": mask_fg,
            "fg_logits": fg_logits,
        }

        return mask_fg, info


class BackgroundnessHead(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        grid_size: int = 14,
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        self.bg_predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        patch_features_last: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        B, N, D = patch_features_last.shape

        bg_logit = self.bg_predictor(patch_features_last).squeeze(-1)
        bg_score = torch.sigmoid(bg_logit)

        uncertainty = self.uncertainty_predictor(patch_features_last).squeeze(-1)

        mask_bg = bg_score * (1.0 - foreground_mask) * uncertainty

        info = {
            "bg_raw_score": bg_score.detach(),
            "uncertainty": uncertainty.detach(),
            "bg_ratio": mask_bg.sum(dim=1).detach() / (mask_bg.sum(dim=1) + foreground_mask.sum(dim=1) + 1e-8),
        }

        return mask_bg, info
