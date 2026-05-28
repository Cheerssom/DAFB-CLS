import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        return self.bce_logits(pred_mask.float(), gt_mask.float())


class MaskDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_mask.float().reshape(pred_mask.shape[0], -1)
        gt_flat = gt_mask.float().reshape(gt_mask.shape[0], -1)
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedMaskLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = MaskBCELoss()
        self.dice = MaskDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred_mask, gt_mask) + self.dice_weight * self.dice(pred_mask, gt_mask)


def generate_pseudo_mask_from_patch_score(
    patch_score: torch.Tensor,
    grid_size: int,
    method: str = "mean_std",
) -> torch.Tensor:
    B, N = patch_score.shape
    if method == "mean_std":
        mean = patch_score.mean(dim=1, keepdim=True)
        std = patch_score.std(dim=1, keepdim=True)
        mask = (patch_score > (mean + std)).float()
    elif method == "top_k_ratio":
        ratio = 0.3
        k = max(1, int(N * ratio))
        _, indices = patch_score.topk(k, dim=1)
        mask = torch.zeros_like(patch_score)
        mask.scatter_(1, indices, 1.0)
    elif method == "otsu_approx":
        sorted_score, _ = patch_score.sort(dim=1)
        best_thresh = sorted_score[:, N // 2].unsqueeze(1)
        mask = (patch_score > best_thresh).float()
    else:
        raise ValueError(f"Unknown pseudo mask method: {method}")
    return mask
