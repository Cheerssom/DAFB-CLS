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


def compute_corloc(score_map: torch.Tensor, bboxes: torch.Tensor, grid_size: int, patch_size: int) -> bool:
    if bboxes.shape[0] == 0:
        return False

    N = score_map.shape[0]
    mean_score = score_map.mean()
    std_score = score_map.std()
    threshold = mean_score + std_score

    foreground_indices = (score_map > threshold).nonzero(as_tuple=True)[0]
    if len(foreground_indices) == 0:
        foreground_indices = score_map.topk(max(1, N // 4)).indices

    fg_rows = foreground_indices // grid_size
    fg_cols = foreground_indices % grid_size
    fg_x = (fg_cols.float() + 0.5) * patch_size
    fg_y = (fg_rows.float() + 0.5) * patch_size

    fg_center_x = fg_x.mean()
    fg_center_y = fg_y.mean()

    for bbox in bboxes:
        if bbox[0] <= fg_center_x <= bbox[2] and bbox[1] <= fg_center_y <= bbox[3]:
            return True

    return False


def evaluate_corloc(model, dataloader, cfg, device):
    model.eval()
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating CorLoc"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            bboxes_list = batch.get("bboxes")
            if bboxes_list is None:
                continue

            output = model(images)
            fg_mask = output["foreground_mask"]

            B = fg_mask.shape[0]
            for i in range(B):
                bbox = bboxes_list[i] if isinstance(bboxes_list, list) else bboxes_list[i]
                if isinstance(bbox, torch.Tensor) and bbox.dim() >= 1 and bbox.shape[0] == 0:
                    continue

                if compute_corloc(fg_mask[i], bbox if isinstance(bbox, torch.Tensor) else torch.tensor(bbox), grid_size, patch_size):
                    correct += 1
                total += 1

    corloc = correct / max(total, 1)
    print(f"CorLoc: {corloc:.4f} ({correct}/{total})")
    return corloc


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
            return_bbox=True,
        )
    elif dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_bbox=True,
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

    evaluate_corloc(model, val_loader, cfg, device)


if __name__ == "__main__":
    main()
