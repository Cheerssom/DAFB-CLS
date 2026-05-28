"""
Nature-style depth attention visualization for DAFB-CLS paper.
Generates publication-quality grouped bar charts of foreground/background
depth attention weights across layers.

Usage:
    python visualizations/plot_depth_attention_nature.py \
        --config dafb_cls/configs/dino_vits16_voc.yaml \
        --checkpoint checkpoints/dino_vits16_voc/final.pt \
        --num_images 4
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ── Nature style ────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ── Color palette: restrained, accessible ───────────────────
C_FG = "#CC6677"       # muted coral red for foreground
C_BG = "#88CCEE"       # muted blue for background
C_ACCENT = "#44AA99"   # teal accent
C_NEUTRAL = "#BBBBBB"  # neutral grey
C_GRID = "#E8E8E8"     # light grid


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_attention_data(model, dataset, device, num_images=4, layer_indices=None):
    """Run model on images and collect depth attention weights."""
    model.eval()
    results = []
    with torch.no_grad():
        for idx in range(min(len(dataset), num_images)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            image_id = sample.get("image_id", f"img_{idx}")
            output = model(image)
            beta_F = output["beta_F"][0].cpu().numpy()
            beta_B = output["beta_B"][0].cpu().numpy()
            results.append({
                "image_id": image_id,
                "beta_F": beta_F,
                "beta_B": beta_B,
            })
    return results


def plot_depth_attention_nature(results, layer_indices, save_dir):
    """Generate Nature-style depth attention figures."""
    os.makedirs(save_dir, exist_ok=True)
    n_layers = len(layer_indices)
    x = np.arange(n_layers)
    layer_labels = [f"L{i}" for i in layer_indices]
    bar_width = 0.32

    # ── Figure 1: Multi-image overview (2x2 grid) ──────────
    n_img = len(results)
    ncols = min(n_img, 2)
    nrows = (n_img + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.08, 2.2 * nrows),
                             sharey=True, squeeze=False)

    for i, res in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        fg = ax.bar(x - bar_width / 2, res["beta_F"], bar_width,
                    color=C_FG, edgecolor="white", linewidth=0.4,
                    label="Foreground", zorder=3)
        bg = ax.bar(x + bar_width / 2, res["beta_B"], bar_width,
                    color=C_BG, edgecolor="white", linewidth=0.4,
                    label="Background", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Attention weight" if i % ncols == 0 else "")
        ax.set_title(res["image_id"], fontsize=7, fontweight="bold", pad=6)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        # direct label for first panel only
        if i == 0:
            ax.legend(loc="upper left", fontsize=5.5, handlelength=1.2,
                      handletextpad=0.4, columnspacing=0.8)

    # hide unused axes
    for j in range(n_img, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Depth-wise attention weights across ViT layers",
                 fontsize=8, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_multi = os.path.join(save_dir, "depth_attention_overview")
    fig.savefig(f"{path_multi}.pdf")
    fig.savefig(f"{path_multi}.png")
    plt.close(fig)
    print(f"  Saved: {path_multi}.pdf / .png")

    # ── Figure 2: Averaged across all images (hero panel) ──
    mean_F = np.mean([r["beta_F"] for r in results], axis=0)
    mean_B = np.mean([r["beta_B"] for r in results], axis=0)
    std_F = np.std([r["beta_F"] for r in results], axis=0)
    std_B = np.std([r["beta_B"] for r in results], axis=0)

    fig2, ax2 = plt.subplots(figsize=(3.2, 2.4))
    fg2 = ax2.bar(x - bar_width / 2, mean_F, bar_width, yerr=std_F,
                  color=C_FG, edgecolor="white", linewidth=0.4,
                  label="Foreground stream", capsize=2, error_kw={"linewidth": 0.5},
                  zorder=3)
    bg2 = ax2.bar(x + bar_width / 2, mean_B, bar_width, yerr=std_B,
                  color=C_BG, edgecolor="white", linewidth=0.4,
                  label="Background stream", capsize=2, error_kw={"linewidth": 0.5},
                  zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("ViT layer")
    ax2.set_ylabel("Depth attention weight")
    ax2.yaxis.grid(True, color=C_GRID, linewidth=0.3, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=6, handlelength=1.2, handletextpad=0.4)

    # annotate dominant layer for each stream
    i_fg = np.argmax(mean_F)
    i_bg = np.argmax(mean_B)
    ax2.annotate(f"peak: L{layer_indices[i_fg]}",
                 xy=(x[i_fg] - bar_width / 2, mean_F[i_fg]),
                 xytext=(x[i_fg] - bar_width / 2 + 0.5, mean_F[i_fg] + 0.08),
                 fontsize=5, color=C_FG, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_FG, lw=0.5))
    ax2.annotate(f"peak: L{layer_indices[i_bg]}",
                 xy=(x[i_bg] + bar_width / 2, mean_B[i_bg]),
                 xytext=(x[i_bg] + bar_width / 2 + 0.5, mean_B[i_bg] + 0.08),
                 fontsize=5, color=C_BG, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_BG, lw=0.5))

    plt.tight_layout()
    path_avg = os.path.join(save_dir, "depth_attention_mean")
    fig2.savefig(f"{path_avg}.pdf")
    fig2.savefig(f"{path_avg}.png")
    plt.close(fig2)
    print(f"  Saved: {path_avg}.pdf / .png")

    # ── Figure 3: Per-image small multiples (for supplement) ──
    ncols3 = min(n_img, 4)
    nrows3 = (n_img + ncols3 - 1) // ncols3
    fig3, axes3 = plt.subplots(nrows3, ncols3,
                               figsize=(7.08, 1.6 * nrows3),
                               sharey=True, squeeze=False)

    for i, res in enumerate(results):
        ax = axes3[i // ncols3][i % ncols3]
        ax.bar(x - bar_width / 2, res["beta_F"], bar_width,
               color=C_FG, edgecolor="white", linewidth=0.3, zorder=3)
        ax.bar(x + bar_width / 2, res["beta_B"], bar_width,
               color=C_BG, edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=5.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(res["image_id"], fontsize=6, pad=3)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.2, zorder=0)
        ax.set_axisbelow(True)
        if i % ncols3 == 0:
            ax.set_ylabel("Weight", fontsize=6)

    for j in range(n_img, nrows3 * ncols3):
        axes3[j // ncols3][j % ncols3].set_visible(False)

    plt.tight_layout()
    path_multi3 = os.path.join(save_dir, "depth_attention_multiples")
    fig3.savefig(f"{path_multi3}.pdf")
    fig3.savefig(f"{path_multi3}.png")
    plt.close(fig3)
    print(f"  Saved: {path_multi3}.pdf / .png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--output_dir", default="visualizations/depth_attention")
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

    layer_indices = cfg.get("layer_indices", [3, 6, 9, 12])
    print(f"Collecting depth attention from {args.num_images} images...")
    results = collect_attention_data(model, dataset, device, args.num_images, layer_indices)

    print("Generating Nature-style depth attention figures...")
    plot_depth_attention_nature(results, layer_indices, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
