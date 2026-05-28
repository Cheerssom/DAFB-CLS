import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.models.cues import FrequencyStabilityCue
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)


def apply_colormap(gray: np.ndarray, cmap: str = "jet") -> np.ndarray:
    from matplotlib import colormaps
    cm_func = colormaps.get_cmap(cmap)
    if gray.ndim == 1:
        grid_size = int(np.sqrt(gray.shape[0]))
        gray = gray.reshape(grid_size, grid_size)
    colored = cm_func(gray / 255.0)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def save_visualization(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray,
    fg_mask_np: np.ndarray,
    bg_mask_np: np.ndarray,
    freq_mask_np: np.ndarray,
    save_path: str,
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask_np, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(apply_colormap(normalize_to_uint8(freq_mask_np)))
    axes[2].set_title("LaSt-ViT Frequency Mask")
    axes[2].axis("off")

    axes[3].imshow(apply_colormap(normalize_to_uint8(fg_mask_np)))
    axes[3].set_title("DAFB Foreground Mask")
    axes[3].axis("off")

    axes[4].imshow(apply_colormap(normalize_to_uint8(bg_mask_np)))
    axes[4].set_title("DAFB Background Mask")
    axes[4].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="visualizations/masks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DAFBCLS(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model = model.to(device)

    freq_cue = FrequencyStabilityCue(low_pass_sigma=cfg.get("frequency_sigma", 0.25))

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    else:
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0

    model.eval()
    with torch.no_grad():
        for idx in range(min(len(val_ds), args.num_images)):
            sample = val_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample.get("mask")
            image_id = sample.get("image_id", f"img_{idx}")

            output = model(image)
            patch_features, _ = model.extractor(image, return_cls=True)
            freq_score = freq_cue(patch_features)

            fg_mask = output["foreground_mask"][0].cpu().numpy()
            bg_mask = output.get("background_mask", output["foreground_mask"])[0].cpu().numpy()
            freq_mask = freq_score[0, -1, :].cpu().numpy()

            grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
            image_np = sample["image"].permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)

            gt_mask_np = gt_mask.numpy() if gt_mask is not None else np.zeros((grid_size, grid_size))

            save_path = os.path.join(args.output_dir, f"{image_id}_masks.png")
            save_visualization(image_np, gt_mask_np, fg_mask, bg_mask, freq_mask, save_path)
            count += 1

    print(f"Saved {count} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
