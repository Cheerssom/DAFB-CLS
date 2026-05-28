import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .feature_extractor import MultiLayerFeatureExtractor
from .dual_cls import DualCLSAggregator
from .depth_attention import ForegroundBackgroundDepthAttention
from .fusion import TaskAdaptiveFusionHead
from .heads import ClassificationHead, SegmentationHead, ObjectDiscoveryScoringHead


class DAFBCLS(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        backbone_type = cfg.get("backbone_type", "openclip_vitb16")
        layer_indices = cfg.get("layer_indices", [3, 6, 9, 12])
        image_size = cfg.get("image_size", 224)
        patch_size = cfg.get("patch_size", 16)
        grid_size = image_size // patch_size

        self.ablation = cfg.get("ablation", {})
        self.disable_dafb = self.ablation.get("disable_dafb", False)
        self.lastvit_mode = self.ablation.get("lastvit_mode", "dual")
        share_depth = self.ablation.get("share_depth_attn", False)

        self.extractor = MultiLayerFeatureExtractor(
            backbone_type=backbone_type,
            layer_indices=layer_indices,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.extractor.build_backbone()
        dim = self.extractor.dim

        self.dual_cls = DualCLSAggregator(
            dim=dim,
            grid_size=grid_size,
            hidden_dim=cfg.get("hidden_dim", 256),
            frequency_sigma=cfg.get("frequency_sigma", 0.25),
            depth_method=cfg.get("depth_method", "cosine"),
            semantic_method=cfg.get("semantic_method", "patch_cls_similarity"),
            spatial_kernel=cfg.get("spatial_kernel", 3),
            tau_from_global=cfg.get("tau_from_global", True),
            ablation=self.ablation,
        )

        self.depth_attn = ForegroundBackgroundDepthAttention(
            dim=dim,
            zero_init=cfg.get("depth_attn_zero_init", True),
            share=share_depth,
        )

        self.fusion = TaskAdaptiveFusionHead(
            dim=dim,
            hidden_dim=cfg.get("fusion_hidden_dim", 256),
        )

        self.task = cfg.get("task", "classification")
        num_classes = cfg.get("num_classes", 20)

        self.text_dim = None
        self.seg_bg_scale = None

        if backbone_type.startswith("openclip"):
            import open_clip
            self.text_dim = 512 if "vitb16" in backbone_type else 768

        if self.task == "classification":
            self.head = ClassificationHead(dim, num_classes, hidden_dim=cfg.get("head_hidden_dim", 512))
        elif self.task == "segmentation":
            self.head = SegmentationHead(
                dim, grid_size=grid_size,
                upsample_size=cfg.get("upsample_size", None),
                text_dim=self.text_dim,
            )
            self.seg_bg_scale = nn.Parameter(torch.tensor(0.0))
            init_scale = cfg.get("logit_scale_init", 2.66)
            self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        elif self.task == "object_discovery":
            self.head = ObjectDiscoveryScoringHead(dim, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def freeze_backbone(self):
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)

    def get_trainable_params(self):
        trainable = []
        for name, p in self.named_parameters():
            if "extractor" not in name:
                trainable.append(p)
        return trainable

    def forward(
        self,
        x: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        patch_features, cls_features = self.extractor(x, return_cls=True)

        if self.disable_dafb:
            last_layer = patch_features[:, -1, :, :]
            global_feat = cls_features if cls_features is not None else last_layer.mean(dim=1)

            if self.lastvit_mode == "cls":
                C = global_feat.unsqueeze(1).expand_as(last_layer)
            else:
                C = last_layer

            output = {
                "C": C.mean(dim=1),
                "C_F": C.mean(dim=1),
                "C_B": C.mean(dim=1),
                "gate": torch.zeros(x.shape[0], device=x.device),
                "beta_F": torch.zeros(x.shape[0], device=x.device),
                "beta_B": torch.zeros(x.shape[0], device=x.device),
                "foreground_mask": torch.ones(x.shape[0], patch_features.shape[2], device=x.device),
                "background_mask": torch.zeros(x.shape[0], patch_features.shape[2], device=x.device),
                "foreground_score": torch.ones(x.shape[0], patch_features.shape[2], device=x.device) * 0.5,
                "fg_logits": torch.zeros(x.shape[0], patch_features.shape[2], device=x.device),
                "patch_features_last": last_layer,
                "agg_info": {},
            }
        else:
            B_F, B_B, agg_info = self.dual_cls(
                patch_features, cls_features=cls_features, text_features=text_features,
            )

            C_F, C_B, beta_F, beta_B = self.depth_attn(B_F, B_B)

            global_feat = cls_features if cls_features is not None else patch_features[:, -1, :, :].mean(dim=1)
            C, gate = self.fusion(C_F, C_B, global_feat)

            output = {
                "C": C,
                "C_F": C_F,
                "C_B": C_B,
                "gate": gate,
                "beta_F": beta_F,
                "beta_B": beta_B,
                "foreground_mask": agg_info["foreground_mask"],
                "background_mask": agg_info.get("background_mask"),
                "foreground_score": agg_info.get("foreground_score"),
                "fg_logits": agg_info.get("fg_logits"),
                "patch_features_last": patch_features[:, -1, :, :],
                "agg_info": agg_info,
            }

        C_out = output["C"]

        if self.task == "classification":
            logits = self.head(C_out)
            output["logits"] = logits
        elif self.task == "segmentation":
            seg_logits = self.head(patch_features[:, -1, :, :], text_features=text_features, cls_feature=C_out)
            if self.seg_bg_scale is not None:
                fg_mask = output["foreground_mask"]
                seg_h = int(seg_logits.shape[1] ** 0.5)
                grid_h = int(fg_mask.shape[1] ** 0.5)
                if seg_h != grid_h:
                    fg_mask = F.interpolate(
                        fg_mask.reshape(fg_mask.shape[0], 1, grid_h, grid_h),
                        size=seg_h, mode="bilinear", align_corners=False,
                    ).reshape(fg_mask.shape[0], -1)
                bg_bias = torch.zeros_like(seg_logits)
                bg_bias[..., 0] = (1 - fg_mask) * self.seg_bg_scale
                seg_logits = seg_logits + bg_bias
            scale = self.logit_scale.exp().clamp(max=100.0)
            seg_logits = seg_logits * scale
            output["seg_logits"] = seg_logits
        elif self.task == "object_discovery":
            fg_mask = output["foreground_mask"]
            score_map = self.head(patch_features[:, -1, :, :], foreground_mask=fg_mask, cls_feature=C_out)
            output["score_map"] = score_map

        return output
