import os

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math


class MultiLayerFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_type: str = "openclip_vitb16",
        layer_indices: Optional[List[int]] = None,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.backbone = None
        self._layer_indices = layer_indices
        self._hooks = []
        self._features: Dict[int, torch.Tensor] = {}

    def build_backbone(self):
        if self.backbone_type == "openclip_vitb16":
            self._build_openclip("ViT-B-16", self._layer_indices or [3, 6, 9, 12])
        elif self.backbone_type == "openclip_vitl14":
            self._build_openclip("ViT-L-14", self._layer_indices or [6, 12, 18, 24])
        elif self.backbone_type == "dino_vits16":
            self._build_dino("vit_small_patch16_224", self._layer_indices or [3, 6, 9, 12])
        elif self.backbone_type == "dino_vitb16":
            self._build_dino("vit_base_patch16_224", self._layer_indices or [3, 6, 9, 12])
        elif self.backbone_type in ("deit_vits16", "deit_vitb16"):
            model_name = "deit_small_patch16_224" if "small" in self.backbone_type else "deit_base_patch16_224"
            self._build_timm(model_name, self._layer_indices or [3, 6, 9, 12])
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type}")

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        return self

    def _build_openclip(self, model_name: str, layer_indices: List[int]):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b88k"
        )
        self.backbone = model.visual
        self._clip_text_encoder = model.encode_text
        if hasattr(model.visual, "ln_post"):
            self._clip_ln_post = model.visual.ln_post
        else:
            self._clip_ln_post = None
        if hasattr(model.visual, "proj") and model.visual.proj is not None:
            self._clip_visual_proj = model.visual.proj.data.clone()
        else:
            self._clip_visual_proj = None
        self._layer_indices = layer_indices
        self._dim = self.backbone.transformer.width
        self._register_vit_hooks()

    def _build_dino(self, model_name: str, layer_indices: List[int]):
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone = model
        self._layer_indices = layer_indices
        self._dim = model.embed_dim
        self._register_timm_hooks(model_name)

    def _build_timm(self, model_name: str, layer_indices: List[int]):
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone = model
        self._layer_indices = layer_indices
        self._dim = model.embed_dim
        self._register_timm_hooks(model_name)

    def _register_vit_hooks(self):
        self._clear_hooks()
        blocks = self.backbone.transformer.resblocks
        for idx in self._layer_indices:
            layer = blocks[idx - 1]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _register_timm_hooks(self, model_name: str):
        self._clear_hooks()
        blocks = self.backbone.blocks
        for idx in self._layer_indices:
            layer = blocks[idx - 1]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output
            self._features[layer_idx] = feat.detach()
        return hook_fn

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def layer_indices(self) -> List[int]:
        return self._layer_indices

    def forward(
        self, x: torch.Tensor, return_cls: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._features.clear()
        _ = self.backbone(x)

        sorted_indices = sorted(self._layer_indices)
        patch_feats = []
        cls_feat = None

        for idx in sorted_indices:
            feat = self._features[idx]
            if feat.shape[1] == self.num_patches + 1:
                if cls_feat is None:
                    cls_feat = feat[:, 0, :]
                patch_feats.append(feat[:, 1:, :])
            elif feat.shape[1] == self.num_patches:
                patch_feats.append(feat)
            else:
                B = feat.shape[0]
                if feat.shape[1] > self.num_patches:
                    if cls_feat is None and return_cls:
                        cls_feat = feat[:, 0, :]
                    patch_feats.append(feat[:, 1:self.num_patches + 1, :])
                else:
                    patch_feats.append(feat[:, :self.num_patches, :])

        patch_features = torch.stack(patch_feats, dim=1)

        if cls_feat is None and return_cls:
            cls_feat = torch.zeros(x.shape[0], self._dim, device=x.device)

        return patch_features, cls_feat


def interpolate_pos_embed(pos_embed, old_grid, new_grid):
    if old_grid == new_grid:
        return pos_embed
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    C = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, C).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, C)
    return torch.cat([cls_pos, patch_pos], dim=1)
