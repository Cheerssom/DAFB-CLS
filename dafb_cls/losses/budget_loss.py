import torch
import torch.nn as nn


class BudgetRegularizationLoss(nn.Module):
    def __init__(
        self,
        r_min: float = 0.1,
        r_max: float = 0.7,
        method: str = "range_penalty",
    ):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.method = method

    def forward(self, foreground_mask: torch.Tensor) -> torch.Tensor:
        r = foreground_mask.mean(dim=1)

        if self.method == "range_penalty":
            loss_min = torch.clamp(self.r_min - r, min=0.0) ** 2
            loss_max = torch.clamp(r - self.r_max, min=0.0) ** 2
            return (loss_min + loss_max).mean()
        elif self.method == "entropy":
            p = foreground_mask.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean(dim=1)
            target_entropy = 0.5
            return ((entropy - target_entropy) ** 2).mean()
        else:
            raise ValueError(f"Unknown budget method: {self.method}")
