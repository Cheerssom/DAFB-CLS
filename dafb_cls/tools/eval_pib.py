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
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset


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


def compute_patch_score(patch_features: torch.Tensor, cls_feature: torch.Tensor) -> torch.Tensor:
    B, N, D = patch_features.shape
    cls_norm = F.normalize(cls_feature, dim=-1).unsqueeze(1)
    patch_norm = F.normalize(patch_features, dim=-1)
    return (patch_norm * cls_norm).sum(dim=-1)


def check_point_in_box(pred_point: torch.Tensor, bboxes: torch.Tensor) -> bool:
    if bboxes.shape[0] == 0:
        return False
    px, py = pred_point[0].item(), pred_point[1].item()
    for bbox in bboxes:
        if bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]:
            return True
    return False


def evaluate_pib(model, dataloader, cfg, device):
    model.eval()
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)
    total = 0
    hits = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating PiB"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            bboxes_list = batch.get("bboxes")
            if bboxes_list is None:
                continue

            output = model(images)
            C = output["C"]
            patch_feat = output["patch_features_last"]

            patch_score = compute_patch_score(patch_feat, C)

            B = patch_score.shape[0]
            for i in range(B):
                bbox = bboxes_list[i] if isinstance(bboxes_list, list) else bboxes_list[i]
                if isinstance(bbox, torch.Tensor) and bbox.dim() >= 1 and bbox.shape[0] == 0:
                    continue

                score_map = patch_score[i]
                max_idx = score_map.argmax().item()
                row = max_idx // grid_size
                col = max_idx % grid_size
                center_x = (col + 0.5) * patch_size
                center_y = (row + 0.5) * patch_size

                if check_point_in_box(
                    torch.tensor([center_x, center_y]),
                    bbox if isinstance(bbox, torch.Tensor) else torch.tensor(bbox),
                ):
                    hits += 1
                total += 1

    pib = hits / max(total, 1)
    print(f"Point-in-Box: {pib:.4f} ({hits}/{total})")
    return pib


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
            return_bbox=True,
        )
    elif dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    evaluate_pib(model, val_loader, cfg, device)


if __name__ == "__main__":
    main()
