import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def visualize_depth_attention(
    beta_F: np.ndarray,
    beta_B: np.ndarray,
    layer_indices: list,
    image_id: str,
    save_path: str,
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_labels = [f"L{i}" for i in layer_indices]
    x = np.arange(len(layer_indices))

    axes[0].bar(x, beta_F, color="red", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_title(f"Foreground Depth Attention - {image_id}")
    axes[0].set_ylabel("Attention Weight")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, beta_B, color="blue", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_title(f"Background Depth Attention - {image_id}")
    axes[1].set_ylabel("Attention Weight")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="visualizations/depth_attention")
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
    layer_indices = cfg.get("layer_indices", [3, 6, 9, 12])
    count = 0

    model.eval()
    with torch.no_grad():
        for idx in range(min(len(val_ds), args.num_images)):
            sample = val_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            image_id = sample.get("image_id", f"img_{idx}")

            output = model(image)
            beta_F = output["beta_F"][0].cpu().numpy()
            beta_B = output["beta_B"][0].cpu().numpy()

            save_path = os.path.join(args.output_dir, f"{image_id}_depth_attn.png")
            visualize_depth_attention(beta_F, beta_B, layer_indices, image_id, save_path)
            count += 1

    print(f"Saved {count} depth attention visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
