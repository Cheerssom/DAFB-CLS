import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .foregroundness import ForegroundnessHead
from .adaptive_budget import AdaptiveBudgetModule, BackgroundnessHead
from .cues import FrequencyStabilityCue, DepthConsistencyCue, SemanticAlignmentCue, SpatialCompactnessCue


class DualCLSAggregator(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_size: int = 14,
        hidden_dim: int = 256,
        frequency_sigma: float = 0.25,
        depth_method: str = "cosine",
        semantic_method: str = "patch_cls_similarity",
        spatial_kernel: int = 3,
        tau_from_global: bool = True,
        ablation: dict = None,
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_patches = grid_size ** 2

        ablation = ablation or {}
        self.disable_cues = ablation.get("disable_cues", False)
        self.disable_adaptive_budget = ablation.get("disable_adaptive_budget", False)
        self.disable_dual_cls = ablation.get("disable_dual_cls", False)
        self.budget_fixed_ratio = ablation.get("budget_fixed_ratio", 0.3)
        mask_type = ablation.get("mask_type", "soft")
        topk_ratio = ablation.get("topk_ratio", 0.3)

        use_texture_complement = ablation.get("use_texture_complement", False)
        texture_weight_init = ablation.get("texture_weight_init", 0.3)
        self.freq_cue = FrequencyStabilityCue(
            low_pass_sigma=frequency_sigma,
            use_texture_complement=use_texture_complement,
            texture_weight_init=texture_weight_init,
        )
        self.depth_cue = DepthConsistencyCue(method=depth_method)
        self.semantic_cue = SemanticAlignmentCue(method=semantic_method)
        self.spatial_cue = SpatialCompactnessCue(kernel_size=spatial_kernel)

        self.foregroundness = ForegroundnessHead(
            dim=dim,
            num_cues=4,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
        )

        temperature_init = ablation.get("temperature_init", 0.1)
        use_score_stats = ablation.get("use_score_stats", False)
        self.budget = AdaptiveBudgetModule(
            dim=dim,
            hidden_dim=hidden_dim // 2,
            tau_from_global=tau_from_global,
            mask_type=mask_type,
            topk_ratio=topk_ratio,
            temperature_init=temperature_init,
            use_score_stats=use_score_stats,
        )

        if not self.disable_dual_cls:
            self.backgroundness = BackgroundnessHead(
                dim=dim,
                hidden_dim=hidden_dim // 2,
                grid_size=grid_size,
            )

    def forward(
        self,
        patch_features: torch.Tensor,
        cls_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B, L, N, D = patch_features.shape

        if self.disable_cues:
            freq_last = torch.ones(B, N, device=patch_features.device) * 0.5
            depth_last = torch.ones(B, N, device=patch_features.device) * 0.5
            semantic_last = torch.ones(B, N, device=patch_features.device) * 0.5
            spatial_last = torch.ones(B, N, device=patch_features.device) * 0.5
        else:
            freq_score = self.freq_cue(patch_features)
            depth_score = self.depth_cue(patch_features)
            semantic_score = self.semantic_cue(patch_features, cls_features=cls_features, text_features=text_features)

            freq_last = freq_score[:, -1, :] if freq_score.dim() == 3 else freq_score
            depth_last = depth_score
            semantic_last = semantic_score
            spatial_last = self.spatial_cue(semantic_last, self.grid_size, self.grid_size)

        F_score, fg_info = self.foregroundness(freq_last, depth_last, semantic_last, spatial_last)

        global_feat = cls_features if cls_features is not None else patch_features[:, -1, :, :].mean(dim=1)

        if self.disable_adaptive_budget:
            k = max(1, int(N * self.budget_fixed_ratio))
            _, topk_indices = F_score.topk(k, dim=1)
            mask_fg = torch.zeros_like(F_score)
            mask_fg.scatter_(1, topk_indices, 1.0)
            budget_info = {
                "tau": torch.zeros(B, device=F_score.device),
                "temperature": torch.ones(1, device=F_score.device),
                "fg_ratio": mask_fg.mean(dim=1).detach(),
                "raw_mask": mask_fg,
                "fg_logits": torch.zeros_like(F_score),
            }
        else:
            mask_fg, budget_info = self.budget(F_score, global_feature=global_feat)

        patch_last = patch_features[:, -1, :, :]
        if self.disable_dual_cls:
            mask_bg = 1.0 - mask_fg
            bg_info = {
                "fg_map": torch.zeros(B, 1, self.grid_size, self.grid_size, device=patch_features.device),
                "bg_map": torch.ones(B, 1, self.grid_size, self.grid_size, device=patch_features.device),
            }
        else:
            mask_bg, bg_info = self.backgroundness(patch_last, mask_fg)

        fg_sum = mask_fg.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
        bg_sum = mask_bg.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)

        mask_fg_expanded = mask_fg.unsqueeze(-1)
        mask_bg_expanded = mask_bg.unsqueeze(-1)

        B_F_list = []
        B_B_list = []
        for l in range(L):
            b_f = (patch_features[:, l, :, :] * mask_fg_expanded).sum(dim=1, keepdim=True) / fg_sum
            b_b = (patch_features[:, l, :, :] * mask_bg_expanded).sum(dim=1, keepdim=True) / bg_sum
            B_F_list.append(b_f.squeeze(1))
            B_B_list.append(b_b.squeeze(1))

        B_F = torch.stack(B_F_list, dim=1)
        B_B = torch.stack(B_B_list, dim=1)

        info = {
            **fg_info,
            **budget_info,
            **bg_info,
            "foreground_mask": mask_fg.detach(),
            "background_mask": mask_bg.detach(),
            "foreground_score": F_score.detach(),
            "fg_logits": budget_info["fg_logits"],
        }

        return B_F, B_B, info
