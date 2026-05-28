import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from scipy import stats

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.datasets.voc import VOCDataset, VOC_CLASSES
from dafb_cls.datasets.coco import COCODataset
from dafb_cls.tools.train_posthoc import collate_fn


STRESS_SUBSETS = {
    "stable_background": {
        "description": "Images with low foreground ratio (<20% GT mask)",
        "max_fg_ratio": 0.20,
    },
    "textured_foreground": {
        "description": "Animals and people (bird, sheep, cow, horse, person, cat, dog)",
        "voc_classes": ["bird", "sheep", "cow", "horse", "person", "cat", "dog"],
    },
    "small_object": {
        "description": "Objects < 5% of image area (in 224x224 space)",
        "max_area_ratio": 0.05,
    },
    "multi_object": {
        "description": "Images with >= 3 object instances",
        "min_instances": 3,
    },
}
IMAGE_AREA = 224 * 224


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def filter_stable_background(dataset):
    """Images where GT foreground mask covers < max_fg_ratio of the image."""
    max_fg_ratio = STRESS_SUBSETS["stable_background"]["max_fg_ratio"]
    indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample.get("mask")
        if mask is not None:
            fg_ratio = (mask > 0).float().mean().item()
            if fg_ratio < max_fg_ratio:
                indices.append(i)
        elif sample.get("bboxes") is not None and len(sample["bboxes"]) > 0:
            bbox_area_sum = sum(
                (b[2] - b[0]) * (b[3] - b[1]) for b in sample["bboxes"]
            )
            fg_ratio = bbox_area_sum / IMAGE_AREA
            if fg_ratio < max_fg_ratio:
                indices.append(i)
    return indices


def filter_textured_foreground(dataset, subset_cfg):
    indices = []
    target_classes = set(c.lower() for c in subset_cfg["voc_classes"])
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample.get("labels", [])
        if any(str(l).lower().strip() in target_classes for l in labels):
            indices.append(i)
    return indices


def filter_small_object(dataset, subset_cfg):
    indices = []
    max_area_ratio = subset_cfg.get("max_area_ratio", 0.05)
    for i in range(len(dataset)):
        sample = dataset[i]
        bboxes = sample.get("bboxes")
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area / IMAGE_AREA < max_area_ratio:
                    indices.append(i)
                    break
    return indices


def filter_multi_object(dataset, subset_cfg):
    indices = []
    min_instances = subset_cfg.get("min_instances", 3)
    for i in range(len(dataset)):
        sample = dataset[i]
        bboxes = sample.get("bboxes")
        if bboxes is not None and len(bboxes) >= min_instances:
            indices.append(i)
    return indices


def compute_stress_metrics(model, dataloader, cfg, device):
    model.eval()
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)

    fg_mask_ious = []
    pib_hits = 0
    pib_total = 0
    pred_fg_ratios = []
    gt_fg_ratios = []
    fg_depth_attn_stats = []
    bg_depth_attn_stats = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Stress Test"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            gt_masks = batch.get("mask")
            bboxes_list = batch.get("bboxes")

            output = model(images)
            fg_mask = output["foreground_mask"]
            beta_F = output["beta_F"]
            beta_B = output["beta_B"]

            B = fg_mask.shape[0]
            for i in range(B):
                pred_ratio = fg_mask[i].mean().item()
                pred_fg_ratios.append(pred_ratio)

                if gt_masks is not None:
                    gt_i = gt_masks[i]
                    if isinstance(gt_i, torch.Tensor):
                        gt_i = gt_i.to(device)
                    if gt_i.dim() == 2:
                        gt_binary = (gt_i > 0).float()
                        gt_ratio = gt_binary.mean().item()
                        gt_fg_ratios.append(gt_ratio)

                        pred_2d = fg_mask[i].reshape(grid_size, grid_size)
                        pred_full = F.interpolate(
                            pred_2d.unsqueeze(0).unsqueeze(0),
                            size=(image_size, image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                        pred_binary = (pred_full > 0.5).float()
                        intersection = (pred_binary * gt_binary).sum()
                        union = pred_binary.sum() + gt_binary.sum() - intersection
                        if union > 0:
                            fg_mask_ious.append((intersection / union).item())

                if bboxes_list is not None and bboxes_list[i].shape[0] > 0:
                    score_map = fg_mask[i]
                    max_idx = score_map.argmax().item()
                    row = max_idx // grid_size
                    col = max_idx % grid_size
                    cx = (col + 0.5) * patch_size
                    cy = (row + 0.5) * patch_size
                    hit = False
                    for bbox in bboxes_list[i]:
                        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                            hit = True
                            break
                    if hit:
                        pib_hits += 1
                    pib_total += 1

                fg_depth_attn_stats.append(beta_F[i].cpu().numpy())
                bg_depth_attn_stats.append(beta_B[i].cpu().numpy())

    metrics = {
        "mean_fg_mask_iou": np.mean(fg_mask_ious) if fg_mask_ious else 0.0,
        "pib": pib_hits / max(pib_total, 1),
        "pib_count": f"{pib_hits}/{pib_total}",
    }

    if len(pred_fg_ratios) > 1 and len(gt_fg_ratios) > 1 and len(pred_fg_ratios) == len(gt_fg_ratios):
        pearson_r, pearson_p = stats.pearsonr(pred_fg_ratios, gt_fg_ratios)
        spearman_r, spearman_p = stats.spearmanr(pred_fg_ratios, gt_fg_ratios)
        metrics["pearson_r"] = pearson_r
        metrics["pearson_p"] = pearson_p
        metrics["spearman_r"] = spearman_r
        metrics["spearman_p"] = spearman_p

    if fg_depth_attn_stats:
        fg_stats = np.array(fg_depth_attn_stats)
        bg_stats = np.array(bg_depth_attn_stats)
        metrics["fg_depth_attn_mean"] = fg_stats.mean(axis=0).tolist()
        metrics["fg_depth_attn_std"] = fg_stats.std(axis=0).tolist()
        metrics["fg_depth_attn_entropy"] = stats.entropy(fg_stats.mean(axis=0) + 1e-8).item()
        metrics["bg_depth_attn_mean"] = bg_stats.mean(axis=0).tolist()
        metrics["bg_depth_attn_std"] = bg_stats.std(axis=0).tolist()
        metrics["bg_depth_attn_entropy"] = stats.entropy(bg_stats.mean(axis=0) + 1e-8).item()

    metrics["mean_pred_fg_ratio"] = np.mean(pred_fg_ratios) if pred_fg_ratios else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None, choices=list(STRESS_SUBSETS.keys()))
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DAFBCLS(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model = model.to(device)

    subsets_to_run = [args.subset] if args.subset else list(STRESS_SUBSETS.keys())

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        full_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        full_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )

    for subset_name in subsets_to_run:
        subset_cfg = STRESS_SUBSETS[subset_name]
        print(f"\n{'='*60}")
        print(f"Stress Subset: {subset_name}")
        print(f"Description: {subset_cfg['description']}")
        print(f"{'='*60}")

        if subset_name == "stable_background":
            indices = filter_stable_background(full_ds)
        elif subset_name == "textured_foreground":
            indices = filter_textured_foreground(full_ds, subset_cfg)
        elif subset_name == "small_object":
            indices = filter_small_object(full_ds, subset_cfg)
        elif subset_name == "multi_object":
            indices = filter_multi_object(full_ds, subset_cfg)
        else:
            indices = list(range(len(full_ds)))

        if len(indices) == 0:
            print(f"  No samples found for {subset_name}, skipping.")
            continue

        subset_ds = Subset(full_ds, indices)
        loader = DataLoader(
            subset_ds,
            batch_size=cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_fn,
        )

        print(f"  Subset size: {len(subset_ds)}")
        metrics = compute_stress_metrics(model, loader, cfg, device)

        print(f"  Foreground Mask IoU: {metrics['mean_fg_mask_iou']:.4f}")
        print(f"  Point-in-Box: {metrics['pib']} ({metrics['pib_count']})")
        print(f"  Mean Predicted FG Ratio: {metrics['mean_pred_fg_ratio']:.4f}")
        if "pearson_r" in metrics:
            print(f"  Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
            print(f"  Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
        print(f"  FG Depth Attn Entropy: {metrics.get('fg_depth_attn_entropy', 'N/A')}")
        print(f"  BG Depth Attn Entropy: {metrics.get('bg_depth_attn_entropy', 'N/A')}")


if __name__ == "__main__":
    main()
