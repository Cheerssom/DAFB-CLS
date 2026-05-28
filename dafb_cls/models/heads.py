import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ClassificationHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self.head(cls_feature)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_size: int = 14,
        method: str = "patch_text_similarity",
        upsample_size: Optional[int] = None,
        text_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.method = method
        self.upsample_size = upsample_size
        out_dim = text_dim if text_dim is not None else dim
        self.proj = nn.Linear(dim, out_dim)
        self.cls_proj = nn.Linear(dim, out_dim)

    def forward(
        self,
        patch_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        cls_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = patch_features.shape
        patch_proj = self.proj(patch_features)
        patch_proj = F.normalize(patch_proj, dim=-1)

        if text_features is not None and self.method == "patch_text_similarity":
            text_norm = F.normalize(text_features, dim=-1)
            similarity = torch.einsum("bnd,kd->bnk", patch_proj, text_norm)
        elif cls_feature is not None:
            cls_proj = self.cls_proj(cls_feature)
            cls_norm = F.normalize(cls_proj, dim=-1).unsqueeze(1)
            similarity = (patch_proj * cls_norm).sum(dim=-1, keepdim=True)
        else:
            raise ValueError("Need text_features or cls_feature")

        if self.upsample_size and self.upsample_size != self.grid_size:
            B_s = similarity.shape[0]
            C_s = similarity.shape[-1]
            sim_2d = similarity.permute(0, 2, 1).reshape(B_s, C_s, self.grid_size, self.grid_size)
            sim_2d = F.interpolate(sim_2d, size=self.upsample_size, mode="bilinear", align_corners=False)
            similarity = sim_2d.reshape(B_s, C_s, -1).permute(0, 2, 1)

        return similarity


class ObjectDiscoveryScoringHead(nn.Module):
    def __init__(self, dim: int, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.score_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
        cls_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = patch_features.shape
        score = self.score_proj(patch_features).squeeze(-1)

        if cls_feature is not None:
            cls_norm = F.normalize(cls_feature, dim=-1).unsqueeze(1)
            patch_norm = F.normalize(patch_features, dim=-1)
            cosine_sim = (patch_norm * cls_norm).sum(dim=-1)
            score = score + cosine_sim

        if foreground_mask is not None:
            score = score * foreground_mask

        return score
