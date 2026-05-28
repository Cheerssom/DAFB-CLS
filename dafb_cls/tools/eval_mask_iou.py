import os
import sys

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.datasets.voc import VOCDataset, VOC_CLASSES
from dafb_cls.datasets.coco import COCODataset, COCO_CLASSES


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            if all(v.shape == values[0].shape for v in values):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        else:
            result[key] = values
    return result


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> dict:
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    iou = intersection / (union + 1e-6)
    precision = intersection / (pred_binary.sum() + 1e-6)
    recall = intersection / (gt_binary.sum() + 1e-6)

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


def encode_text_features(cfg, device):
    backbone_type = cfg.get("backbone_type", "")
    if not backbone_type.startswith("openclip"):
        return None
    import open_clip
    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        classes = list(COCO_CLASSES) + ["background"]
    else:
        classes = list(VOC_CLASSES) + ["background"]
    model_name = "ViT-B-16" if "vitb16" in backbone_type else "ViT-L-14"
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b88k")
    clip_model = clip_model.to(device)
    clip_model.eval()
    texts = [f"a photo of a {c}" for c in classes]
    tokenizer = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenizer)
        text_features = F.normalize(text_features, dim=-1)
    del clip_model
    torch.cuda.empty_cache()
    return text_features


def evaluate_mask_iou(model, dataloader, cfg, device, text_features=None):
    model.eval()
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)

    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Mask IoU"):
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

            output = model(images, text_features=text_features)
            fg_mask = output["foreground_mask"]

            B = fg_mask.shape[0]
            for i in range(B):
                pred_2d = fg_mask[i].reshape(grid_size, grid_size)
                pred_full = F.interpolate(
                    pred_2d.unsqueeze(0).unsqueeze(0),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                gt_i = gt_masks[i]
                if gt_i.max() > 1:
                    gt_i = (gt_i > 0).float()

                metrics = compute_mask_iou(pred_full, gt_i)
                all_iou.append(metrics["iou"])
                all_precision.append(metrics["precision"])
                all_recall.append(metrics["recall"])

    mean_iou = np.mean(all_iou) if all_iou else 0.0
    mean_precision = np.mean(all_precision) if all_precision else 0.0
    mean_recall = np.mean(all_recall) if all_recall else 0.0

    print(f"Mask IoU: {mean_iou:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")

    return {
        "mean_iou": mean_iou,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DAFBCLS(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model = model.to(device)

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "voc":
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    elif dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    text_features = encode_text_features(cfg, device)
    if text_features is not None:
        print(f"Encoded text features: {text_features.shape}")

    evaluate_mask_iou(model, val_loader, cfg, device, text_features=text_features)


if __name__ == "__main__":
    main()
