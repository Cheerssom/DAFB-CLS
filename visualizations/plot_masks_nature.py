"""
Nature-style mask visualization for DAFB-CLS paper.
Generates publication-quality 5-panel figures: Input / GT Mask / FG Mask / BG Mask / Foregroundness Score.

Usage:
    python visualizations/plot_masks_nature.py \
        --config dafb_cls/configs/dino_vits16_voc.yaml \
        --checkpoint checkpoints/dino_vits16_voc/final.pt \
        --num_images 4
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── Nature style ────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ── Colors ──────────────────────────────────────────────────
C_FG = "#CC6677"
C_BG = "#88CCEE"
C_ACCENT = "#44AA99"
C_NEUTRAL = "#BBBBBB"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized image tensor back to displayable numpy array."""
    img = tensor.cpu().float()
    # handle both normalized [~(-2,2)] and raw uint8 [0,255] inputs
    if img.max() > 10:
        img = img / 255.0
    else:
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def overlay_mask(image, mask, color, alpha=0.45):
    """Overlay a colored mask on an image."""
    overlay = image.copy()
    color_arr = np.array(mcolors.to_rgb(color))
    for c in range(3):
        overlay[:, :, c] = image[:, :, c] * (1 - alpha * mask) + color_arr[c] * alpha * mask
    return overlay


def collect_mask_data(model, dataset, device, num_images=4):
    """Run model on images and collect mask predictions."""
    model.eval()
    results = []
    with torch.no_grad():
        for idx in range(min(len(dataset), num_images)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            image_id = sample.get("image_id", f"img_{idx}")

            output = model(image)
            fg_mask = output["foreground_mask"][0].cpu().numpy()
            bg_mask = output["background_mask"][0].cpu().numpy() if output.get("background_mask") is not None else (1 - fg_mask)
            fg_score = output["foreground_score"][0].cpu().numpy() if output.get("foreground_score") is not None else fg_mask
            beta_F = output["beta_F"][0].cpu().numpy()
            beta_B = output["beta_B"][0].cpu().numpy()

            gt_mask = sample.get("mask", None)
            if gt_mask is not None:
                gt_mask = gt_mask.cpu().numpy()

            results.append({
                "image_id": image_id,
                "image": sample["image"],
                "fg_mask": fg_mask,
                "bg_mask": bg_mask,
                "fg_score": fg_score,
                "gt_mask": gt_mask,
                "beta_F": beta_F,
                "beta_B": beta_B,
            })
    return results


def grid_size_from_flat(flat_size):
    """Infer spatial grid size from flat mask length."""
    h = int(np.sqrt(flat_size))
    return h, flat_size // h


def plot_mask_panel_nature(results, save_dir):
    """Generate Nature-style mask visualization figures."""
    os.makedirs(save_dir, exist_ok=True)

    for res in results:
        image_id = res["image_id"]
        img = denormalize_image(res["image"])
        H, W = img.shape[:2]
        fg = res["fg_mask"]
        bg = res["bg_mask"]
        score = res["fg_score"]
        gt = res["gt_mask"]

        gh, gw = grid_size_from_flat(len(fg))

        # resize masks to image size
        def to_img(mask_flat):
            m = mask_flat.reshape(gh, gw)
            m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            m_up = F.interpolate(m_t, size=(H, W), mode="bilinear", align_corners=False)
            return m_up.squeeze().numpy()

        fg_up = to_img(fg)
        bg_up = to_img(bg)
        score_up = to_img(score)

        # create 5-panel figure
        fig = plt.figure(figsize=(7.08, 1.6))
        gs = GridSpec(1, 5, figure=fig, wspace=0.15,
                      left=0.02, right=0.98, top=0.88, bottom=0.05)

        panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

        # Panel (a): Input image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(img)
        ax0.set_title("Input", fontsize=6.5, fontweight="bold")
        ax0.axis("off")

        # Panel (b): GT mask overlay
        ax1 = fig.add_subplot(gs[1])
        if gt is not None:
            gt_up = to_img(gt) if gt.ndim == 1 else gt
            if gt_up.shape != (H, W):
                gt_t = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                gt_up = F.interpolate(gt_t, size=(H, W), mode="nearest").squeeze().numpy()
            ax1.imshow(overlay_mask(img, gt_up, "#44AA99", alpha=0.5))
        else:
            ax1.imshow(img * 0.3 + 0.35)
        ax1.set_title("Ground Truth", fontsize=6.5, fontweight="bold")
        ax1.axis("off")

        # Panel (c): Foreground mask overlay
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(overlay_mask(img, fg_up, C_FG, alpha=0.5))
        ax2.set_title("Foreground", fontsize=6.5, fontweight="bold", color=C_FG)
        ax2.axis("off")

        # Panel (d): Background mask overlay
        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(overlay_mask(img, bg_up, C_BG, alpha=0.5))
        ax3.set_title("Background", fontsize=6.5, fontweight="bold", color=C_BG)
        ax3.axis("off")

        # Panel (e): Foregroundness score heatmap
        ax4 = fig.add_subplot(gs[4])
        cmap = mpl.colormaps.get_cmap("RdYlBu_r")
        im = ax4.imshow(score_up, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        ax4.set_title("Score", fontsize=6.5, fontweight="bold")
        ax4.axis("off")
        # compact colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.55])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("$F_i$", fontsize=6, labelpad=2)
        cbar.ax.tick_params(labelsize=5)

        # panel labels
        for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4]):
            ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                    fontsize=6, fontweight="bold", va="top", ha="left",
                    color="white", bbox=dict(boxstyle="round,pad=0.15",
                    facecolor="black", alpha=0.5, edgecolor="none"))

        path = os.path.join(save_dir, f"{image_id}_mask_panel")
        fig.savefig(f"{path}.pdf")
        fig.savefig(f"{path}.png")
        plt.close(fig)
        print(f"  Saved: {path}.pdf / .png")

    # ── Summary figure: grid of all images (input + FG overlay) ──
    n = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols * 2, figsize=(7.08, 1.8 * nrows),
                               squeeze=False)

    for i, res in enumerate(results):
        img = denormalize_image(res["image"])
        H, W = img.shape[:2]
        fg_up = to_img(res["fg_mask"])
        col = i % ncols
        row = i // ncols

        ax_img = axes2[row][col * 2]
        ax_fg = axes2[row][col * 2 + 1]

        ax_img.imshow(img)
        ax_img.set_title(res["image_id"], fontsize=5.5, pad=2)
        ax_img.axis("off")

        ax_fg.imshow(overlay_mask(img, fg_up, C_FG, alpha=0.5))
        ax_fg.set_title("FG mask", fontsize=5.5, pad=2, color=C_FG)
        ax_fg.axis("off")

    # hide unused
    for j in range(n, nrows * ncols):
        col = j % ncols
        row = j // ncols
        axes2[row][col * 2].set_visible(False)
        axes2[row][col * 2 + 1].set_visible(False)

    plt.tight_layout()
    path_grid = os.path.join(save_dir, "mask_overview_grid")
    fig2.savefig(f"{path_grid}.pdf")
    fig2.savefig(f"{path_grid}.png")
    plt.close(fig2)
    print(f"  Saved: {path_grid}.pdf / .png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--output_dir", default="visualizations/masks")
    args = parser.parse_args()

    from dafb_cls.models.dafb_cls_model import DAFBCLS
    from dafb_cls.datasets.voc import VOCDataset
    from dafb_cls.datasets.coco import COCODataset

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DAFBCLS(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model.to(device)

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        dataset = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann",
                             os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    else:
        dataset = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )

    print(f"Collecting mask predictions from {args.num_images} images...")
    results = collect_mask_data(model, dataset, device, args.num_images)

    print("Generating Nature-style mask figures...")
    plot_mask_panel_nature(results, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
