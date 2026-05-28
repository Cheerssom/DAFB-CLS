"""
DINO-seg baseline: Use DINO self-attention maps for object discovery.

DINO-seg (Caron et al. 2021):
  1. Extract self-attention maps from the last ViT layer
  2. Average attention from CLS token to all patches across all heads
  3. Threshold or Top-K to get foreground mask
  4. Evaluate CorLoc, Mask IoU, PiB

Usage:
    python -m dafb_cls.tools.eval_dinoseg --config dafb_cls/configs/dino_vits16_voc.yaml
    python -m dafb_cls.tools.eval_dinoseg --config dafb_cls/configs/dino_vits16_coco.yaml
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
from dafb_cls.tools.train_posthoc import collate_fn


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DINOSegBaseline:
    """DINO self-attention baseline for object discovery."""

    def __init__(self, backbone_type="dino_vits16", last_layer=True):
        import timm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = {
            "dino_vits16": "vit_small_patch16_224",
            "dino_vitb16": "vit_base_patch16_224",
        }.get(backbone_type, "vit_small_patch16_224")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Hook to capture attention weights
        self.attention_weights = None
        self._register_attention_hook(last_layer)

    def _register_attention_hook(self, last_layer=True):
        """Register hook on the last attention block to capture attention weights."""
        blocks = self.model.blocks
        target_block = blocks[-1]  # last layer

        def attention_hook(module, input, output):
            # timm ViT attention: output is the attention output
            # We need to capture the attention weights
            pass

        # For timm ViT, we need to hook into the attention module's forward
        # to capture the attention weights before softmax
        attn = target_block.attn

        def attn_hook(module, args, output):
            # In timm, attention.forward returns (attn_output, attn_weights)
            # But by default attn_weights are not returned
            # We need to manually compute them
            pass

        # Alternative: register a hook on the full block
        def block_hook(module, input, output):
            # input[0] is the hidden states
            # We'll compute attention manually
            pass

        self._target_block = target_block
        self._hook_handles = []

    def get_attention_map(self, images):
        """Get CLS-to-patch attention map from the last layer."""
        with torch.no_grad():
            # Get the block input
            x = self.model.patch_embed(images)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.model.pos_embed

            # Forward through all blocks except the last
            for blk in self.model.blocks[:-1]:
                x = blk(x)

            # Last block: compute attention manually
            last_blk = self.model.blocks[-1]
            norm1 = last_blk.norm1(x)
            qkv = last_blk.attn.qkv(norm1).reshape(
                x.shape[0], x.shape[1], 3, last_blk.attn.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention scores
            scale = q.shape[-1] ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N+1, N+1]
            attn = attn.softmax(dim=-1)

            # CLS token attention to patches (skip position 0 which is CLS itself)
            # attn shape: [B, heads, N+1, N+1]
            cls_attn = attn[:, :, 0, 1:]  # [B, heads, N]
            cls_attn = cls_attn.mean(dim=1)  # average over heads [B, N]

            return cls_attn

    def get_foreground_mask(self, images, method="threshold"):
        """Get foreground mask from attention map."""
        attn_map = self.get_attention_map(images)  # [B, N]
        B, N = attn_map.shape

        if method == "threshold":
            # Mean + 1 std threshold
            mean = attn_map.mean(dim=1, keepdim=True)
            std = attn_map.std(dim=1, keepdim=True)
            mask = (attn_map > mean + std).float()
        elif method == "topk":
            # Top 30%
            k = max(1, int(N * 0.3))
            _, indices = attn_map.topk(k, dim=1)
            mask = torch.zeros_like(attn_map)
            mask.scatter_(1, indices, 1.0)
        elif method == "softmax":
            # Direct softmax attention as soft mask
            mask = attn_map / attn_map.sum(dim=1, keepdim=True)
            mask = mask * N  # normalize to ~1 range
        else:
            mask = attn_map

        return mask, attn_map


def evaluate_corloc_dinoseg(model, dataloader, device):
    """Evaluate CorLoc using DINO-seg attention."""
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

        mask, attn_map = model.get_foreground_mask(images, method="threshold")
        B, N = mask.shape
        grid_size = int(N ** 0.5)

        # Find center of attention
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

            # Weighted center of attention
            w = mask[i]
            cx = (w * xx).sum() / (w.sum() + 1e-8)
            cy = (w * yy).sum() / (w.sum() + 1e-8)

            # Convert to pixel coordinates
            img_size = 224
            cx_px = cx.item() / grid_size * img_size
            cy_px = cy.item() / grid_size * img_size

            # Check if center is in any GT bbox
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


def evaluate_mask_iou_dinoseg(model, dataloader, device):
    """Evaluate Mask IoU using DINO-seg attention."""
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

        mask, attn_map = model.get_foreground_mask(images, method="threshold")
        B, N = mask.shape
        grid_size = int(N ** 0.5)

        # Resize pred mask to image size
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


def evaluate_pib_dinoseg(model, dataloader, device):
    """Evaluate Point-in-Box using DINO-seg attention."""
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

        _, attn_map = model.get_foreground_mask(images, method="softmax")
        B, N = attn_map.shape
        grid_size = int(N ** 0.5)

        # Find max attention patch
        max_idx = attn_map.argmax(dim=1)  # [B]
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

    # Load DINO model
    backbone_type = cfg.get("backbone_type", "dino_vits16")
    model = DINOSegBaseline(backbone_type=backbone_type)

    print(f"\n{'='*60}")
    print(f"DINO-seg Baseline Evaluation ({dataset_type.upper()})")
    print(f"{'='*60}")

    # CorLoc
    corloc, correct, total = evaluate_corloc_dinoseg(model, val_loader, device)
    print(f"CorLoc: {corloc:.4f} ({correct}/{total})")

    # Mask IoU
    miou, mprec, mrec, count = evaluate_mask_iou_dinoseg(model, val_loader, device)
    print(f"Mask IoU: {miou:.4f}  (Precision: {mprec:.4f}, Recall: {mrec:.4f})")

    # PiB
    pib, pib_correct, pib_total = evaluate_pib_dinoseg(model, val_loader, device)
    print(f"PiB: {pib:.4f} ({pib_correct}/{pib_total})")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  CorLoc:   {corloc*100:.2f}%")
    print(f"  Mask IoU: {miou*100:.2f}%")
    print(f"  PiB:      {pib*100:.2f}%")


if __name__ == "__main__":
    main()
