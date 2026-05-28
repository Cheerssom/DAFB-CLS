"""
Evaluate LaSt-ViT (frequency stability + hard top-K) as a baseline.

LaSt-ViT algorithm (from conf.py, Shi et al. 2026):
  1. FFT along feature dim D
  2. Gaussian low-pass filter
  3. Stability = x / |x - x_lp|
  4. mean(stability, dim=-1) → per-patch scores
  5. Top-K patches → CLS token (avg of selected patches)

Metrics: CorLoc, Mask IoU, PiB.
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
from collections import OrderedDict

from dafb_cls.models.feature_extractor import MultiLayerFeatureExtractor
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset
from dafb_cls.tools.train_posthoc import collate_fn


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """LaSt-ViT: 1D Gaussian kernel in frequency domain."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.max()


def compute_lastvit_patch_scores(
    patch_features: torch.Tensor,  # [B, N, D] — last-layer patch tokens
    sigma: float = None,
) -> torch.Tensor:
    """
    LaSt-ViT frequency-stability per-patch scores.

    Returns [B, N] where higher = more stable (= more foreground in LaSt-ViT).
    """
    B, N, D = patch_features.shape
    if sigma is None:
        sigma = D ** 0.5  # LaSt-ViT default

    x_detach = patch_features.float()

    # FFT along feature dimension
    x_fft = torch.fft.fft(x_detach, dim=-1)

    # Gaussian low-pass filter
    gk = gaussian_kernel_1d(D, sigma).to(x_fft.device)
    x_fft = torch.fft.fftshift(x_fft, dim=-1)
    x_fft = x_fft * gk.unsqueeze(0).unsqueeze(0)
    x_fft = torch.fft.ifftshift(x_fft, dim=-1)
    x_lp = torch.fft.ifft(x_fft, dim=-1).real  # [B, N, D]

    # Stability: original / |original - lowpass|
    diff = x_detach / (torch.abs(x_lp - x_detach) + 1e-8)  # [B, N, D]

    # Per-patch score: mean stability across feature dims
    patch_scores = diff.mean(dim=-1)  # [B, N]
    return patch_scores


def compute_lastvit_cls(
    patch_features: torch.Tensor,
    sigma: float = None,
) -> torch.Tensor:
    """LaSt-ViT CLS token: select top-1 per dim, average."""
    B, N, D = patch_features.shape
    if sigma is None:
        sigma = D ** 0.5

    x_detach = patch_features.float()
    x_fft = torch.fft.fft(x_detach, dim=-1)
    gk = gaussian_kernel_1d(D, sigma).to(x_fft.device)
    x_fft = torch.fft.fftshift(x_fft, dim=-1)
    x_fft = x_fft * gk.unsqueeze(0).unsqueeze(0)
    x_fft = torch.fft.ifftshift(x_fft, dim=-1)
    x_lp = torch.fft.ifft(x_fft, dim=-1).real

    diff = x_detach / (torch.abs(x_lp - x_detach) + 1e-8)  # [B, N, D]

    # Top-1 per dimension (dim=1 is patch dim, k=1)
    _, indices = torch.topk(diff, k=1, dim=1, largest=True)  # [B, 1, D]
    sel_p = torch.gather(x_detach, 1, indices.expand(-1, -1, D))  # [B, 1, D]
    cls_token = torch.mean(sel_p, dim=1)  # [B, D]
    return cls_token


def evaluate_corloc_lastvit(extractor, dataloader, cfg, device):
    """CorLoc using LaSt-ViT stability scores as foregroundness."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CorLoc"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            bboxes_list = batch.get("bboxes")

            patch_feats, _ = extractor(images, return_cls=False)
            patch_last = patch_feats[:, -1, :, :]  # [B, N, D]
            scores = compute_lastvit_patch_scores(patch_last)  # [B, N]

            B = scores.shape[0]
            for i in range(B):
                bboxes = bboxes_list[i] if isinstance(bboxes_list, list) else bboxes_list[i]
                if hasattr(bboxes, 'shape') and bboxes.shape[0] == 0:
                    continue

                s = scores[i]
                mean_s = s.mean()
                std_s = s.std()
                threshold = mean_s + std_s
                fg_idx = (s > threshold).nonzero(as_tuple=True)[0]
                if len(fg_idx) == 0:
                    fg_idx = s.topk(max(1, s.shape[0] // 4)).indices

                fg_rows = fg_idx // grid_size
                fg_cols = fg_idx % grid_size
                fg_cx = (fg_cols.float() + 0.5).mean() * patch_size
                fg_cy = (fg_rows.float() + 0.5).mean() * patch_size

                hit = False
                for bbox in bboxes:
                    if bbox[0] <= fg_cx <= bbox[2] and bbox[1] <= fg_cy <= bbox[3]:
                        hit = True
                        break
                if hit:
                    correct += 1
                total += 1

    corloc = correct / max(total, 1)
    print(f"LaSt-ViT CorLoc: {corloc:.4f} ({correct}/{total})")
    return corloc


def evaluate_mask_iou_lastvit(extractor, dataloader, cfg, device, topk_ratio=0.3):
    """Mask IoU using LaSt-ViT top-K stability patches as foreground."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)
    N = grid_size ** 2

    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
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

            patch_feats, _ = extractor(images, return_cls=False)
            patch_last = patch_feats[:, -1, :, :]
            scores = compute_lastvit_patch_scores(patch_last)  # [B, N]

            B = scores.shape[0]
            k = max(1, int(N * topk_ratio))
            for i in range(B):
                _, topk_idx = scores[i].topk(k)
                mask = torch.zeros(N, device=device)
                mask.scatter_(0, topk_idx, 1.0)

                pred_2d = mask.reshape(grid_size, grid_size).float()
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
    print(f"LaSt-ViT Mask IoU: {mean_iou:.4f}  (Precision: {mean_p:.4f}, Recall: {mean_r:.4f})")
    return {"mean_iou": mean_iou, "mean_precision": mean_p, "mean_recall": mean_r}


def evaluate_pib_lastvit(extractor, dataloader, cfg, device):
    """PiB using LaSt-ViT CLS token vs patch similarity."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    hits = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="PiB"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            bboxes_list = batch.get("bboxes")
            if bboxes_list is None:
                continue

            patch_feats, _ = extractor(images, return_cls=False)
            patch_last = patch_feats[:, -1, :, :]  # [B, N, D]
            cls_lv = compute_lastvit_cls(patch_last)  # [B, D]

            # Patch most similar to LaSt-ViT CLS token
            cls_norm = F.normalize(cls_lv, dim=-1).unsqueeze(1)
            patch_norm = F.normalize(patch_last, dim=-1)
            similarity = (patch_norm * cls_norm).sum(dim=-1)  # [B, N]

            B = similarity.shape[0]
            for i in range(B):
                bboxes = bboxes_list[i] if isinstance(bboxes_list, list) else bboxes_list[i]
                if hasattr(bboxes, 'shape') and bboxes.shape[0] == 0:
                    continue

                max_idx = similarity[i].argmax().item()
                row = max_idx // grid_size
                col = max_idx % grid_size
                cx = (col + 0.5) * patch_size
                cy = (row + 0.5) * patch_size

                hit = False
                for bbox in bboxes:
                    if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                        hit = True
                        break
                if hit:
                    hits += 1
                total += 1

    pib = hits / max(total, 1)
    print(f"LaSt-ViT PiB: {pib:.4f} ({hits}/{total})")
    return pib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Base config (e.g. dino_vits16_voc.yaml)")
    parser.add_argument("--topk-ratio", type=float, default=0.3,
                        help="Top-K ratio for mask (default=0.3, LaSt-ViT paper uses ~0.25)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Backbone: {cfg.get('backbone_type')}")
    print(f"Top-K ratio: {args.topk_ratio}")

    # Build extractor (frozen backbone, no DAFB components)
    extractor = MultiLayerFeatureExtractor(
        backbone_type=cfg.get("backbone_type", "dino_vits16"),
        layer_indices=cfg.get("layer_indices", [3, 6, 9, 12]),
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 16),
    )
    extractor.build_backbone().to(device)
    extractor.eval()

    # Shared dataloader
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
        pin_memory=True, collate_fn=collate_fn,
    )

    print(f"\n{'='*60}")
    print("LaSt-ViT Baseline Evaluation")
    print(f"{'='*60}")

    corloc = evaluate_corloc_lastvit(extractor, val_loader, cfg, device)
    mask_iou = evaluate_mask_iou_lastvit(extractor, val_loader, cfg, device, topk_ratio=args.topk_ratio)
    pib = evaluate_pib_lastvit(extractor, val_loader, cfg, device)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CorLoc:   {corloc*100:.2f}%")
    print(f"  Mask IoU: {mask_iou['mean_iou']*100:.2f}%")
    print(f"  PiB:      {pib*100:.2f}%")


if __name__ == "__main__":
    main()
