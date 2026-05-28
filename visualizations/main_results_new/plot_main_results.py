
"""
Optimized DAFB-CLS Main Results Figure.

This version moves the shared method legend between the two figure rows,
keeps the two rows compact, and centers the x-axis labels in panel d.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Editable vector text in SVG/PDF when possible
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})

# Palette kept from the supplied figure for method consistency
C_DAFB      = "#0F4D92"
C_CAM       = "#9E9E9E"
C_DINOSEG   = "#6D6D6D"
C_LASTVIT   = "#4A4A4A"
C_TOKENCUT  = "#3D8B8B"
C_MASKCLIP  = "#7A7A9A"
C_DAFB_LIGHT = "#6C9BD2"

VOC_DINO = {
    "methods":  ["CAM", "DINO-seg", "LaSt-ViT", "TokenCut", "DAFB-CLS"],
    "CorLoc":   [62.80, 68.46, 65.70, 58.52, 79.43],
    "MaskIoU":  [3.52,  8.83,  13.31, 6.17,  31.27],
    "PiB":      [21.26, 24.64, 48.10, 25.60, 45.07],
}

VOC_CLIP = {
    "methods":  ["MaskCLIP", "DAFB-CLS"],
    "CorLoc":   [74.88, 77.29],
    "MaskIoU":  [4.07,  33.32],
    "PiB":      [39.68, 79.64],
}

COCO_DINO = {
    "methods":  ["CAM", "DINO-seg", "LaSt-ViT", "TokenCut", "DAFB-CLS"],
    "CorLoc":   [59.81, 67.23, 61.53, 54.75, 72.96],
    "MaskIoU":  [4.30,  9.24,  12.95, 6.77,  28.73],
    "PiB":      [24.96, 29.00, 45.09, 31.97, 35.44],
}

COCO_CLIP = {
    "methods":  ["MaskCLIP", "DAFB-CLS"],
    "CorLoc":   [70.56, None],
    "MaskIoU":  [6.05,  None],
    "PiB":      [39.36, None],
}

ABLATION = {
    "short":     ["Base.", "-Cues", "-Budget", "-DualCls", "Shared", "Hard-K", "Full"],
    "CorLoc":    [29.33, 47.48, 79.78, 78.40, 77.43, 81.71, 79.43],
    "MaskIoU":   [30.76, 30.16, 13.91, 31.25, 31.18, 22.68, 31.27],
}

METRICS = ["CorLoc (%)", "Mask IoU (%)", "PiB (%)"]
METRIC_KEYS = ["CorLoc", "MaskIoU", "PiB"]


def method_color(name):
    return {
        "DAFB-CLS": C_DAFB,
        "CAM": C_CAM,
        "DINO-seg": C_DINOSEG,
        "LaSt-ViT": C_LASTVIT,
        "TokenCut": C_TOKENCUT,
        "MaskCLIP": C_MASKCLIP,
    }.get(name, "#888888")


def clean_axes(ax):
    ax.grid(axis="y", alpha=0.22, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def grouped_bar_panel(ax, ds_dino, ds_clip, title):
    methods = list(ds_dino["methods"])
    values = {k: list(ds_dino[k]) for k in METRIC_KEYS}

    for i, method in enumerate(ds_clip["methods"]):
        vals = {k: ds_clip[k][i] for k in METRIC_KEYS}
        if all(v is not None for v in vals.values()):
            methods.append(f"{method} (CLIP)")
            for k in METRIC_KEYS:
                values[k].append(vals[k])

    n_methods = len(methods)
    x = np.arange(len(METRICS), dtype=float)
    group_width = 0.82
    bar_width = group_width / n_methods

    for j, method in enumerate(methods):
        center_shift = (j - (n_methods - 1) / 2) * bar_width
        vals = [values[k][j] for k in METRIC_KEYS]
        bars = ax.bar(
            x + center_shift,
            vals,
            width=bar_width * 0.9,
            color=method_color(method.split(" (")[0]),
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
            label=method,
        )

        for bar, val in zip(bars, vals):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color="#333333",
                    clip_on=False,
                )

    max_val = max(v for vals in values.values() for v in vals if v is not None)
    ax.set_ylim(0, max_val + 10)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=8)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=11)
    clean_axes(ax)
    return methods


def delta_panel(ax):
    methods = ["vs TokenCut", "vs CAM", "vs LaSt-ViT", "vs DINO-seg"]
    base_corr = np.array([58.52, 62.80, 65.70, 68.46])
    base_miou = np.array([6.17, 3.52, 13.31, 8.83])
    d_corr = 79.43 - base_corr
    d_miou = 31.27 - base_miou

    y = np.arange(len(methods))
    h = 0.34
    b1 = ax.barh(
        y + h / 2,
        d_corr,
        height=h,
        color=C_DAFB,
        edgecolor="white",
        linewidth=0.55,
        label=r"$\Delta$ CorLoc",
        zorder=3,
    )
    b2 = ax.barh(
        y - h / 2,
        d_miou,
        height=h,
        color=C_DAFB_LIGHT,
        edgecolor="white",
        linewidth=0.55,
        label=r"$\Delta$ Mask IoU",
        zorder=3,
    )

    for bars, vals, col in [(b1, d_corr, C_DAFB), (b2, d_miou, C_DAFB_LIGHT)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.45,
                bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                color=col,
                fontweight="bold",
                clip_on=False,
            )

    ax.axvline(0, color="#333333", linewidth=0.8, zorder=4)
    ax.set_xlim(-2, 34)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Improvement (pp)", fontsize=9)
    ax.set_title("(c) DAFB-CLS Improvement over Baselines (DINO, VOC)", loc="left", fontsize=11, fontweight="bold", pad=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.8, borderaxespad=0.4)


def ablation_panel(ax):
    idx = [1, 2, 3, 4, 5, 6]
    labels = [ABLATION["short"][i] for i in idx]
    corr = [ABLATION["CorLoc"][i] for i in idx]
    miou = [ABLATION["MaskIoU"][i] for i in idx]

    x = np.arange(len(labels))
    width = 0.36
    b1 = ax.bar(
        x - width / 2,
        corr,
        width=width * 0.92,
        color=C_DAFB,
        edgecolor="white",
        linewidth=0.55,
        label="CorLoc",
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        miou,
        width=width * 0.92,
        color=C_DAFB_LIGHT,
        edgecolor="white",
        linewidth=0.55,
        label="Mask IoU",
        zorder=3,
    )

    for bars, vals, col in [(b1, corr, C_DAFB), (b2, miou, C_DAFB_LIGHT)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=col,
                clip_on=False,
            )

    ax.axvspan(len(labels) - 1 - 0.48, len(labels) - 1 + 0.48, alpha=0.07, color=C_DAFB, zorder=0)
    ax.set_ylim(10, 92)
    ax.set_xlim(-0.65, len(labels) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_title("(d) Component Ablation (DINO, VOC)", loc="left", fontsize=11, fontweight="bold", pad=11)
    ax.text(
        0.985,
        0.965,
        "Baseline DINO: CorLoc 29.3%, Mask IoU 30.8%",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        style="italic",
        va="top",
        ha="right",
    )
    clean_axes(ax)
    ax.legend(loc="upper left", fontsize=8, handlelength=1.8, borderaxespad=0.4)


def main():
    out_dir = "/mnt/data"

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.55))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    methods = grouped_bar_panel(ax_a, VOC_DINO, VOC_CLIP, "(a) VOC 2012 Object Discovery & Segmentation")
    grouped_bar_panel(ax_b, COCO_DINO, COCO_CLIP, "(b) COCO 2017 Cross-Dataset Generalization")
    delta_panel(ax_c)
    ablation_panel(ax_d)

    # A single top legend avoids overlap with x-axis labels and lower-panel titles.
    legend_handles = [
        Patch(
            facecolor=method_color(m.split(" (")[0]),
            edgecolor="white",
            linewidth=0.55,
            label=m,
        )
        for m in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.505),
        ncol=len(methods),
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.15,
        handletextpad=0.45,
    )

    # The shared legend is placed between the upper and lower rows; keep the row spacing compact.
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.955, wspace=0.31, hspace=0.38)

    pdf = os.path.join(out_dir, "main_results_legend_middle_aligned_final.pdf")
    png = os.path.join(out_dir, "main_results_legend_middle_aligned_final.png")
    svg = os.path.join(out_dir, "main_results_legend_middle_aligned_final.svg")
    script = os.path.join(out_dir, "plot_main_results_legend_middle_aligned_final.py")

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=600)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    return pdf, png, svg, script


if __name__ == "__main__":
    pdf, png, svg, script = main()
    print("Saved optimized files:")
    print(pdf)
    print(png)
    print(svg)
    print(script)
