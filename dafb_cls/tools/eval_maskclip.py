"""
MaskCLIP baseline for open-vocabulary object discovery.

MaskCLIP (Zhou et al., 2022):
  1. Extract patch features from CLIP visual encoder (intermediate layers, bypassing final attention pooling)
  2. Compute per-patch similarity with text embeddings for each class
  3. Generate per-patch class predictions via masked attention
  4. Use foreground class predictions as object discovery mask

For object discovery (class-agnostic):
  - Aggregate all non-background class similarities as foregroundness score
  - Evaluate CorLoc, Mask IoU, PiB

Usage:
    python -m dafb_cls.tools.eval_maskclip --config dafb_cls/configs/openclip_vitb16_voc.yaml
    python -m dafb_cls.tools.eval_maskclip --config dafb_cls/configs/openclip_vitb16_coco.yaml
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dafb_cls.datasets.voc import VOCDataset, VOC_CLASSES
from dafb_cls.datasets.coco import COCODataset, COCO_CLASSES
from dafb_cls.tools.train_posthoc import collate_fn


class MaskCLIPBaseline:
    """MaskCLIP: open-vocabulary segmentation via masked CLIP features."""

    def __init__(self, backbone_type="openclip_vitb16"):
        import open_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "vitb16" in backbone_type:
            model_name = "ViT-B-16"
        elif "vitl14" in backbone_type:
            model_name = "ViT-L-14"
        else:
            model_name = "ViT-B-16"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b88k"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.tokenize

    def encode_text(self, class_names):
        """Encode class text prompts."""
        prompts = [f"a photo of a {name}" for name in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features  # [C, D]

    def get_maskclip_features(self, images):
        """
        Extract MaskCLIP patch features from intermediate layers.

        MaskCLIP key insight: bypass the final attention pooling layer and
        use the penultimate layer's patch features, which preserve spatial
        information better than the pooled CLS token.
        """
        with torch.no_grad():
            # Get the visual transformer's intermediate features
            visual = self.model.visual

            # Patch embedding
            x = visual.conv1(images)  # [B, D, H/p, W/p]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, D, N]
            x = x.permute(0, 2, 1)  # [B, N, D]

            # CLS token
            cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
            x = x + visual.positional_embedding.unsqueeze(0)

            # Transformer blocks — use penultimate layer (MaskCLIP strategy)
            # Skip the final attention pooling to preserve per-patch spatial features
            for i, block in enumerate(visual.transformer.resblocks):
                if i == len(visual.transformer.resblocks) - 1:
                    break  # MaskCLIP: skip last block
                x = block(x)

            # Final layernorm
            x = visual.ln_post(x)

            # Remove CLS token, keep patch features
            patch_features = x[:, 1:, :]  # [B, N, D]

            # Project to CLIP embedding space
            if hasattr(visual, 'proj') and visual.proj is not None:
                patch_features = patch_features @ visual.proj  # [B, N, D_proj]

            patch_features = F.normalize(patch_features, dim=-1)

        return patch_features  # [B, N, D]

    def get_per_patch_class_scores(self, images, text_features, bg_scale=0.1):
        """
        Compute per-patch class similarity scores.

        Args:
            images: [B, 3, H, W]
            text_features: [C, D] normalized class text embeddings
            bg_scale: scale factor for background class (index 0 if present)

        Returns:
            scores: [B, N, C] per-patch per-class similarity
        """
        patch_features = self.get_maskclip_features(images)  # [B, N, D]
        # Patch-text cosine similarity
        scores = patch_features @ text_features.T  # [B, N, C]

        return scores

    def get_foreground_mask(self, images, text_features, bg_class_idx=0):
        """
        Get class-agnostic foreground mask from MaskCLIP.

        Foreground = max similarity over all non-background classes.
        """
        scores = self.get_per_patch_class_scores(images, text_features)  # [B, N, C]

        # Separate background and foreground classes
        C = scores.shape[-1]
        if bg_class_idx is not None and C > 1:
            # Background is class 0 (in our setup)
            bg_score = scores[:, :, bg_class_idx:bg_class_idx+1]  # [B, N, 1]
            fg_scores = scores[:, :, 1:]  # [B, N, C-1] — all non-bg classes
            # Foreground score: max over foreground classes
            fg_score = fg_scores.max(dim=-1).values  # [B, N]
            # Contrast with background
            fg_mask = (fg_score - bg_score.squeeze(-1)).sigmoid()  # [B, N]
        else:
            fg_score = scores.max(dim=-1).values  # [B, N]
            fg_mask = (fg_score - fg_score.mean(dim=1, keepdim=True)).sigmoid()

        return fg_mask, scores


def evaluate_corloc_maskclip(model, dataloader, cfg, device, text_features):
    """CorLoc evaluation using MaskCLIP foreground scores."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="CorLoc"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        bboxes = batch.get("bboxes")
        if bboxes is None:
            continue

        fg_mask, _ = model.get_foreground_mask(images, text_features)  # [B, N]
        B = fg_mask.shape[0]

        for i in range(B):
            bbox_list = bboxes[i] if isinstance(bboxes, list) else bboxes[i]
            if hasattr(bbox_list, 'shape') and len(bbox_list.shape) > 0 and bbox_list.shape[0] == 0:
                continue
            if isinstance(bbox_list, torch.Tensor):
                bbox_list = bbox_list.tolist()
            if len(bbox_list) == 0:
                continue

            # Threshold foreground mask
            s = fg_mask[i]
            threshold = 0.5
            fg_idx = (s > threshold).nonzero(as_tuple=True)[0]
            if len(fg_idx) == 0:
                fg_idx = s.topk(max(1, s.shape[0] // 4)).indices

            fg_rows = fg_idx // grid_size
            fg_cols = fg_idx % grid_size
            fg_cx = (fg_cols.float() + 0.5).mean() * patch_size
            fg_cy = (fg_rows.float() + 0.5).mean() * patch_size

            hit = False
            for bbox in bbox_list:
                if isinstance(bbox, (list, tuple)):
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if x1 <= fg_cx <= x2 and y1 <= fg_cy <= y2:
                    hit = True
                    break
            if hit:
                correct += 1
            total += 1

    corloc = correct / max(total, 1)
    print(f"MaskCLIP CorLoc: {corloc:.4f} ({correct}/{total})")
    return corloc


def evaluate_mask_iou_maskclip(model, dataloader, cfg, device, text_features):
    """Mask IoU evaluation using MaskCLIP foreground scores."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)

    all_iou, all_precision, all_recall = [], [], []

    for batch in tqdm(dataloader, desc="Mask IoU"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        gt_masks = batch.get("mask")
        if gt_masks is None:
            continue
        if isinstance(gt_masks, list):
            gt_masks = torch.stack([m.to(device) for m in gt_masks])
        else:
            gt_masks = gt_masks.to(device)

        fg_mask, _ = model.get_foreground_mask(images, text_features)  # [B, N]
        B = fg_mask.shape[0]

        for i in range(B):
            pred_2d = fg_mask[i].reshape(grid_size, grid_size).float()
            pred_full = F.interpolate(
                pred_2d.unsqueeze(0).unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            gt_i = gt_masks[i]
            if gt_i.max() > 1:
                gt_i = (gt_i > 0).float()
            gt_binary = gt_i.float()

            pred_binary = (pred_full > 0.5).float()
            intersection = (pred_binary * gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum() - intersection

            iou = intersection / (union + 1e-6)
            precision = intersection / (pred_binary.sum() + 1e-6)
            recall = intersection / (gt_binary.sum() + 1e-6)
            all_iou.append(iou.item())
            all_precision.append(precision.item())
            all_recall.append(recall.item())

    mean_iou = np.mean(all_iou) if all_iou else 0.0
    mean_p = np.mean(all_precision) if all_precision else 0.0
    mean_r = np.mean(all_recall) if all_recall else 0.0
    print(f"MaskCLIP Mask IoU: {mean_iou:.4f}  (Precision: {mean_p:.4f}, Recall: {mean_r:.4f})")
    return {"mean_iou": mean_iou, "mean_precision": mean_p, "mean_recall": mean_r}


def evaluate_pib_maskclip(model, dataloader, cfg, device, text_features):
    """PiB evaluation using MaskCLIP patch-text similarity."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    hits, total = 0, 0

    for batch in tqdm(dataloader, desc="PiB"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        bboxes = batch.get("bboxes")
        if bboxes is None:
            continue

        fg_mask, scores = model.get_foreground_mask(images, text_features)  # [B, N], [B, N, C]

        # For PiB: find the patch most similar to any foreground class
        # Exclude background class (index 0)
        if scores.shape[-1] > 1:
            fg_scores = scores[:, :, 1:].max(dim=-1).values  # [B, N]
        else:
            fg_scores = scores.squeeze(-1)

        B = fg_scores.shape[0]
        for i in range(B):
            bbox_list = bboxes[i] if isinstance(bboxes, list) else bboxes[i]
            if hasattr(bbox_list, 'shape') and len(bbox_list.shape) > 0 and bbox_list.shape[0] == 0:
                continue
            if isinstance(bbox_list, torch.Tensor):
                bbox_list = bbox_list.tolist()
            if len(bbox_list) == 0:
                continue

            max_idx = fg_scores[i].argmax().item()
            row = max_idx // grid_size
            col = max_idx % grid_size
            cx = (col + 0.5) * patch_size
            cy = (row + 0.5) * patch_size

            hit = False
            for bbox in bbox_list:
                if isinstance(bbox, (list, tuple)):
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1

    pib = hits / max(total, 1)
    print(f"MaskCLIP PiB: {pib:.4f} ({hits}/{total})")
    return pib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML (e.g. openclip_vitb16_voc.yaml)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    backbone_type = cfg.get("backbone_type", "openclip_vitb16")
    dataset_type = cfg.get("dataset", "voc")

    print(f"Device: {device}")
    print(f"Backbone: {backbone_type}")
    print(f"Dataset: {dataset_type}")

    model = MaskCLIPBaseline(backbone_type=backbone_type)

    # Get class names for text encoding
    if dataset_type == "coco":
        classes = COCO_CLASSES
        bg_classes = ["background"] + classes
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        classes = VOC_CLASSES
        bg_classes = ["background"] + classes
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )

    # Encode text features for all classes (including background)
    print(f"Encoding {len(bg_classes)} class text prompts...")
    text_features = model.encode_text(bg_classes)  # [C, D]

    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 16),
        shuffle=False, num_workers=cfg.get("num_workers", 0),
        pin_memory=True, collate_fn=collate_fn,
    )

    print(f"\n{'='*60}")
    print(f"MaskCLIP Baseline Evaluation ({dataset_type.upper()})")
    print(f"{'='*60}")

    corloc = evaluate_corloc_maskclip(model, val_loader, cfg, device, text_features)
    mask_iou = evaluate_mask_iou_maskclip(model, val_loader, cfg, device, text_features)
    pib = evaluate_pib_maskclip(model, val_loader, cfg, device, text_features)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CorLoc:   {corloc*100:.2f}%")
    print(f"  Mask IoU: {mask_iou['mean_iou']*100:.2f}%")
    print(f"  PiB:      {pib*100:.2f}%")


if __name__ == "__main__":
    main()
