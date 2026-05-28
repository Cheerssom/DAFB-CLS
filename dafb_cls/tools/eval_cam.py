"""
CAM/GradCAM baseline for object discovery.

Uses the CLS token's last-layer attention (standard ViT attention map)
as a class-agnostic attention baseline.

This is a simplified version: we extract the attention from CLS to patches
in the last layer, averaged over all attention heads.

Usage:
    python -m dafb_cls.tools.eval_cam --config dafb_cls/configs/dino_vits16_voc.yaml
    python -m dafb_cls.tools.eval_cam --config dafb_cls/configs/dino_vits16_coco.yaml
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

from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset
from dafb_cls.models.feature_extractor import MultiLayerFeatureExtractor
from dafb_cls.tools.train_posthoc import collate_fn


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_cam_scores(extractor, images, method="cls_attn"):
    """
    Compute CAM-like scores for each patch.

    Methods:
      - cls_attn: CLS token attention from last layer (standard ViT attention)
      - cls_sim: cosine similarity between each patch and CLS token
      - norm: L2 norm of each patch feature
    """
    with torch.no_grad():
        patch_features, cls_features = extractor(images)
        # patch_features: [B, L, N, D], cls_features: [B, D]

        # Use last layer
        last_patches = patch_features[:, -1, :, :]  # [B, N, D]

        if method == "cls_attn":
            # Cosine similarity with CLS as proxy for attention
            cls_norm = F.normalize(cls_features, dim=-1)  # [B, D]
            patch_norm = F.normalize(last_patches, dim=-1)  # [B, N, D]
            scores = (patch_norm * cls_norm.unsqueeze(1)).sum(dim=-1)  # [B, N]

        elif method == "cls_sim":
            # Same as cls_attn but explicit
            cls_norm = F.normalize(cls_features, dim=-1)
            patch_norm = F.normalize(last_patches, dim=-1)
            scores = (patch_norm * cls_norm.unsqueeze(1)).sum(dim=-1)

        elif method == "norm":
            # L2 norm of patch features
            scores = last_patches.norm(dim=-1)  # [B, N]

        else:
            raise ValueError(f"Unknown method: {method}")

        return scores


def scores_to_mask(scores, method="threshold"):
    """Convert continuous scores to binary mask."""
    B, N = scores.shape

    if method == "threshold":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True)
        mask = (scores > mean + std).float()
    elif method == "topk":
        k = max(1, int(N * 0.3))
        _, indices = scores.topk(k, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
    else:
        mask = scores

    return mask


def evaluate_corloc_cam(extractor, dataloader, device, method="cls_attn"):
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

        scores = compute_cam_scores(extractor, images, method=method)
        mask = scores_to_mask(scores, method="threshold")
        B, N = mask.shape
        grid_size = int(N ** 0.5)

        grid_y = torch.arange(grid_size, device=device).float()
        grid_x = torch.arange(grid_size, device=device).float()
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        for i in range(B):
            if isinstance(bboxes, list):
                bbox_list = bboxes[i]
                if isinstance(bbox_list, torch.Tensor):
                    bbox_list = bbox_list.tolist()
            else:
                bbox_list = bboxes[i].tolist()

            if len(bbox_list) == 0:
                continue

            w = mask[i]
            cx = (w * xx).sum() / (w.sum() + 1e-8)
            cy = (w * yy).sum() / (w.sum() + 1e-8)

            img_size = 224
            cx_px = cx.item() / grid_size * img_size
            cy_px = cy.item() / grid_size * img_size

            hit = False
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                if x1 <= cx_px <= x2 and y1 <= cy_px <= y2:
                    hit = True
                    break

            if hit:
                correct += 1
            total += 1

    return correct / max(total, 1), correct, total


def evaluate_mask_iou_cam(extractor, dataloader, device, method="cls_attn"):
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Mask IoU"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)

        gt_mask = batch.get("mask")
        if gt_mask is None:
            continue
        if isinstance(gt_mask, list):
            gt_mask = torch.stack([m.to(device) for m in gt_mask])
        else:
            gt_mask = gt_mask.to(device)

        scores = compute_cam_scores(extractor, images, method=method)
        mask = scores_to_mask(scores, method="threshold")
        B, N = mask.shape
        grid_size = int(N ** 0.5)

        pred = mask.reshape(B, 1, grid_size, grid_size)
        pred = F.interpolate(pred, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = (pred.squeeze(1) > 0.5).float()

        gt = (gt_mask > 0).float()

        for i in range(B):
            p = pred[i].reshape(-1)
            g = gt[i].reshape(-1)
            inter = (p * g).sum()
            union = ((p + g) > 0).float().sum()
            if union > 0:
                iou = inter / union
                prec = inter / (p.sum() + 1e-8)
                rec = inter / (g.sum() + 1e-8)
                total_iou += iou.item()
                total_precision += prec.item()
                total_recall += rec.item()
                count += 1

    miou = total_iou / max(count, 1)
    mprec = total_precision / max(count, 1)
    mrec = total_recall / max(count, 1)
    return miou, mprec, mrec, count


def evaluate_pib_cam(extractor, dataloader, device, method="cls_attn"):
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="PiB"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)

        bboxes = batch.get("bboxes")
        if bboxes is None:
            continue

        scores = compute_cam_scores(extractor, images, method=method)
        B, N = scores.shape
        grid_size = int(N ** 0.5)

        max_idx = scores.argmax(dim=1)
        max_y = max_idx // grid_size
        max_x = max_idx % grid_size

        for i in range(B):
            if isinstance(bboxes, list):
                bbox_list = bboxes[i]
                if isinstance(bbox_list, torch.Tensor):
                    bbox_list = bbox_list.tolist()
            else:
                bbox_list = bboxes[i].tolist()

            if len(bbox_list) == 0:
                continue

            img_size = 224
            px = max_x[i].item() / grid_size * img_size
            py = max_y[i].item() / grid_size * img_size

            hit = False
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                if x1 <= px <= x2 and y1 <= py <= y2:
                    hit = True
                    break

            if hit:
                correct += 1
            total += 1

    return correct / max(total, 1), correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="cls_attn",
                        choices=["cls_attn", "cls_sim", "norm"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 16),
        shuffle=False, num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    # Load model
    extractor = MultiLayerFeatureExtractor(
        backbone_type=cfg.get("backbone_type", "openclip_vitb16"),
        layer_indices=cfg.get("layer_indices", [3, 6, 9, 12]),
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 16),
    )
    extractor.build_backbone().to(device)
    extractor.eval()

    print(f"\n{'='*60}")
    print(f"CAM Baseline Evaluation ({dataset_type.upper()}, method={args.method})")
    print(f"{'='*60}")

    # CorLoc
    corloc, correct, total = evaluate_corloc_cam(extractor, val_loader, device, method=args.method)
    print(f"CorLoc: {corloc:.4f} ({correct}/{total})")

    # Mask IoU
    miou, mprec, mrec, count = evaluate_mask_iou_cam(extractor, val_loader, device, method=args.method)
    print(f"Mask IoU: {miou:.4f}  (Precision: {mprec:.4f}, Recall: {mrec:.4f})")

    # PiB
    pib, pib_correct, pib_total = evaluate_pib_cam(extractor, val_loader, device, method=args.method)
    print(f"PiB: {pib:.4f} ({pib_correct}/{pib_total})")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  CorLoc:   {corloc*100:.2f}%")
    print(f"  Mask IoU: {miou*100:.2f}%")
    print(f"  PiB:      {pib*100:.2f}%")


if __name__ == "__main__":
    main()
