"""
Nature-style figure: Ablation studies (DINO & OpenCLIP)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 5.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#CC6677', '#88CCEE', '#44AA99', '#DDCC77', '#AA4499', '#332288', '#BBBBBB']
MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X']

# ── DINO ablation ───────────────────────────────────────────
variants_dino = ['baseline', 'full', 'no_budget', 'no_cues',
                 'no_dual_cls', 'shared_depth', 'hard_topk']
labels_dino = ['Baseline\n(DINO)', 'Full\nDAFB-CLS', 'w/o\nBudget', 'w/o\nCues',
               'w/o Dual\nCLS', 'Shared\nDepth', 'Hard\nTop-K']
corloc_dino  = [29.33, 79.43, 79.78, 47.48, 78.40, 77.43, 81.71]
mask_dino    = [30.76, 31.27, 13.91, 30.16, 31.25, 31.18, 22.68]
pib_dino     = [42.72, 45.07, 53.62, 31.68, 33.13, 28.36, 51.76]

# ── OpenCLIP ablation ───────────────────────────────────────
variants_clip = ['baseline', 'full', 'no_budget', 'no_cues',
                 'no_dual_cls', 'shared_depth', 'hard_topk']
labels_clip = ['Baseline\n(CLIP)', 'Full\nDAFB-CLS', 'w/o\nBudget', 'w/o\nCues',
               'w/o Dual\nCLS', 'Shared\nDepth', 'Hard\nTop-K']
miou_clip = [59.02, 62.27, 64.80, 62.61, 61.65, 62.34, 61.66]

# ── Figure 1: DINO multi-metric grouped bar ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.2))

ax = axes[0]
x = np.arange(len(labels_dino))
width = 0.25
for i, (data, label, color) in enumerate(zip(
    [corloc_dino, mask_dino, pib_dino],
    ['CorLoc (%)', 'Mask IoU (%)', 'PiB (%)'],
    ['#CC6677', '#88CCEE', '#44AA99']
)):
    bars = ax.bar(x + i * width, data, width, label=label, color=color,
                  edgecolor='white', linewidth=0.4)
    if i == 0:  # mark full
        for j, v in enumerate(data):
            if variants_dino[j] == 'full':
                bars[j].set_edgecolor('#333333')
                bars[j].set_linewidth(1.2)

ax.set_xticks(x + width)
ax.set_xticklabels(labels_dino, fontsize=5.5)
ax.set_ylabel('Score (%)')
ax.set_title('Ablation: DINO ViT-S/16 + VOC', fontweight='bold', pad=8)
ax.set_ylim(0, 95)
ax.legend(frameon=False, loc='upper left', fontsize=5.5)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax.set_axisbelow(True)

# ── Figure 2: OpenCLIP mIoU bar ─────────────────────────────
ax2 = axes[1]
x2 = np.arange(len(labels_clip))
bar_colors = ['#BBBBBB'] + ['#CC6677'] + ['#BBBBBB'] * 5
bars2 = ax2.bar(x2, miou_clip, 0.6, color=bar_colors, edgecolor='white', linewidth=0.4)
# Highlight full and no_budget (best)
bars2[1].set_edgecolor('#333333')
bars2[1].set_linewidth(1.2)
bars2[2].set_edgecolor('#333333')
bars2[2].set_linewidth(0.8)
bars2[2].set_linestyle('--')

for bar, val in zip(bars2, miou_clip):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=5.5)

ax2.set_xticks(x2)
ax2.set_xticklabels(labels_clip, fontsize=5.5)
ax2.set_ylabel('mIoU (%)')
ax2.set_title('Ablation: OpenCLIP ViT-B/16 + VOC', fontweight='bold', pad=8)
ax2.set_ylim(55, 70)
ax2.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax2.set_axisbelow(True)

fig.suptitle('Component Ablation Study', fontsize=8.5, fontweight='bold', y=1.02)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'ablation', 'ablation_all.pdf')
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

# ── Figure 3: DINO ablation radar chart ─────────────────────
from matplotlib.patches import FancyBboxPatch

fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
metrics_radar = ['CorLoc', 'Mask IoU', 'PiB']
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

# Normalize to 0-1 scale (divide by max observed)
max_vals = [max(corloc_dino), max(mask_dino), max(pib_dino)]
radar_data = {
    'Full DAFB-CLS': [corloc_dino[1]/max_vals[0], mask_dino[1]/max_vals[1], pib_dino[1]/max_vals[2]],
    'w/o Cues':      [corloc_dino[3]/max_vals[0], mask_dino[3]/max_vals[1], pib_dino[3]/max_vals[2]],
    'w/o Budget':    [corloc_dino[2]/max_vals[0], mask_dino[2]/max_vals[1], pib_dino[2]/max_vals[2]],
    'Baseline':      [corloc_dino[0]/max_vals[0], mask_dino[0]/max_vals[1], pib_dino[0]/max_vals[2]],
}
radar_colors = ['#CC6677', '#88CCEE', '#44AA99', '#BBBBBB']

for (name, vals), color in zip(radar_data.items(), radar_colors):
    vals_closed = vals + vals[:1]
    ax3.plot(angles, vals_closed, '-o', label=name, color=color, linewidth=1.2, markersize=4)
    ax3.fill(angles, vals_closed, alpha=0.08, color=color)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics_radar, fontsize=7)
ax3.set_ylim(0, 1.15)
ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(['25%', '50%', '75%', 'Max'], fontsize=5.5, color='#666666')
ax3.legend(frameon=False, fontsize=5.5, loc='upper right', bbox_to_anchor=(1.3, 1.15))
ax3.set_title('DINO Ablation: Key Variants', fontweight='bold', fontsize=8, pad=15)
fig3.tight_layout()
out3 = os.path.join(os.path.dirname(__file__), 'ablation', 'ablation_radar.pdf')
fig3.savefig(out3, bbox_inches='tight')
fig3.savefig(out3.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved: {out3}')
plt.close()
