import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class FrequencyStabilityCue(nn.Module):
    def __init__(
        self,
        low_pass_sigma: float = 0.25,
        eps: float = 1e-6,
        use_texture_complement: bool = False,
        texture_weight_init: float = 0.3,
    ):
        super().__init__()
        self.low_pass_sigma = low_pass_sigma
        self.eps = eps
        self.use_texture_complement = use_texture_complement

        if use_texture_complement:
            # Learnable weight: 0 = pure stability, 1 = pure texture energy
            self.texture_weight = nn.Parameter(torch.tensor(float(texture_weight_init)))

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        B, L, N, D = patch_features.shape
        x = patch_features.reshape(B * L * N, D).float()

        x_fft = torch.fft.fft(x, dim=-1)
        freqs = torch.fft.fftfreq(D, device=x.device)
        gaussian_lowpass = torch.exp(-0.5 * (freqs / self.low_pass_sigma) ** 2)
        x_lp = x_fft * gaussian_lowpass.unsqueeze(0)
        x_filtered = torch.fft.ifft(x_lp, dim=-1).real

        stability = x_filtered / (torch.abs(x_filtered - x) + self.eps)
        stability = stability.reshape(B, L, N, D)
        stability_score = stability.mean(dim=-1)

        if self.use_texture_complement:
            # High-freq residual norm = per-patch texture energy
            # Smooth patches (sky/wall) → low texture, textured (objects) → high texture
            residual = x.reshape(B, L, N, D) - x_filtered.reshape(B, L, N, D)
            texture_energy = torch.norm(residual, p=2, dim=-1)  # [B, L, N]
            # Normalize per-image to [0, 1] range for stable blending
            tex_flat = texture_energy.reshape(B, -1)
            tex_min = tex_flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
            tex_max = tex_flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
            texture_score = (texture_energy - tex_min) / (tex_max - tex_min + self.eps)

            # Use last-layer values (same as what caller extracts via [:, -1, :])
            texture_last = texture_score[:, -1, :]     # [B, N]
            stability_last = stability_score[:, -1, :]  # [B, N]

            # Learnable blend: w=0 → pure stability, w=1 → pure texture energy
            w = self.texture_weight.sigmoid()
            return w * texture_last + (1 - w) * stability_last

        return stability_score


class DepthConsistencyCue(nn.Module):
    def __init__(self, method: str = "cosine"):
        super().__init__()
        self.method = method

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        B, L, N, D = patch_features.shape

        if L < 2:
            return torch.ones(B, N, device=patch_features.device)

        mean_feat = patch_features.mean(dim=1, keepdim=True)

        if self.method == "cosine":
            norm_feat = F.normalize(patch_features, dim=-1)
            norm_mean = F.normalize(mean_feat, dim=-1)
            sim = (norm_feat * norm_mean).sum(dim=-1)
            consistency = sim.mean(dim=1)
        elif self.method == "variance":
            var = patch_features.var(dim=1).sum(dim=-1)
            consistency = 1.0 / (1.0 + var)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return consistency


class SemanticAlignmentCue(nn.Module):
    def __init__(self, method: str = "patch_cls_similarity"):
        super().__init__()
        self.method = method
        self._fg_prototype = None

    def set_foreground_prototype(self, prototype: torch.Tensor):
        self._fg_prototype = prototype

    def forward(
        self,
        patch_features: torch.Tensor,
        cls_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, N, D = patch_features.shape

        if self.method == "patch_cls_similarity" and cls_features is not None:
            cls_feat = cls_features.unsqueeze(1).unsqueeze(1)
            norm_patch = F.normalize(patch_features, dim=-1)
            norm_cls = F.normalize(cls_feat, dim=-1)
            similarity = (norm_patch * norm_cls).sum(dim=-1)
            return similarity.mean(dim=1)

        elif self.method == "patch_text_similarity" and text_features is not None:
            last_layer_feat = patch_features[:, -1, :, :]
            norm_patch = F.normalize(last_layer_feat, dim=-1)
            norm_text = F.normalize(text_features, dim=-1)
            similarity = torch.einsum("bnd,kd->bnk", norm_patch, norm_text)
            similarity = similarity.max(dim=-1).values
            return similarity

        elif self.method == "patch_prototype" and self._fg_prototype is not None:
            last_layer_feat = patch_features[:, -1, :, :]
            norm_patch = F.normalize(last_layer_feat, dim=-1)
            norm_proto = F.normalize(self._fg_prototype, dim=-1)
            similarity = (norm_patch * norm_proto.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            return similarity

        else:
            last_layer_feat = patch_features[:, -1, :, :]
            mean_feat = last_layer_feat.mean(dim=1, keepdim=True)
            norm_patch = F.normalize(last_layer_feat, dim=-1)
            norm_mean = F.normalize(mean_feat, dim=-1)
            similarity = (norm_patch * norm_mean).sum(dim=-1)
            return similarity


class SpatialCompactnessCue(nn.Module):
    def __init__(self, kernel_size: int = 3, method: str = "avg_pool"):
        super().__init__()
        self.kernel_size = kernel_size
        self.method = method
        if method == "avg_pool":
            self.smoother = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        elif method == "gaussian":
            self._build_gaussian_kernel(kernel_size)

    def _build_gaussian_kernel(self, k: int):
        sigma = k / 3.0
        x = torch.arange(k, dtype=torch.float32) - k // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))
        self.smoother = None

    def forward(self, score_map: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        B = score_map.shape[0]
        score_2d = score_map.reshape(B, 1, grid_h, grid_w)

        if self.method == "avg_pool":
            smoothed = self.smoother(score_2d)
        elif self.method == "gaussian":
            smoothed = F.conv2d(score_2d, self.kernel.to(score_2d.device), padding=self.kernel.shape[-1] // 2)
        else:
            smoothed = score_2d

        return smoothed.reshape(B, -1)
