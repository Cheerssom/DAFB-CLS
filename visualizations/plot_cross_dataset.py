"""
Nature-style figure: Cross-dataset consistency analysis
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

COLORS = ['#CC6677', '#88CCEE', '#44AA99']

# ── VOC vs COCO absolute scores ─────────────────────────────
metrics = ['CorLoc (%)', 'Mask IoU (%)', 'PiB (%)']
voc_las   = [65.70, 13.31, 48.10]
voc_dafb  = [79.43, 31.27, 45.07]
coco_las  = [61.53, 12.95, 45.09]
coco_dafb = [72.96, 28.73, 35.44]

fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.8))

# Panel A: Absolute scores grouped
ax = axes[0]
x = np.arange(len(metrics))
width = 0.2
bars1 = ax.bar(x - 1.5*width, voc_las,  width, label='LaSt-ViT (VOC)',
               color='#BBBBBB', edgecolor='white', linewidth=0.4)
bars2 = ax.bar(x - 0.5*width, voc_dafb, width, label='DAFB-CLS (VOC)',
               color='#CC6677', edgecolor='white', linewidth=0.4)
bars3 = ax.bar(x + 0.5*width, coco_las,  width, label='LaSt-ViT (COCO)',
               color='#DDCC77', edgecolor='white', linewidth=0.4)
bars4 = ax.bar(x + 1.5*width, coco_dafb, width, label='DAFB-CLS (COCO)',
               color='#88CCEE', edgecolor='white', linewidth=0.4)
for bars in [bars2, bars4]:
    for bar in bars:
        bar.set_edgecolor('#333333')
        bar.set_linewidth(0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=6.5)
ax.set_ylabel('Score (%)')
ax.set_title('Absolute Scores', fontweight='bold', fontsize=7.5, pad=6)
ax.set_ylim(0, 95)
ax.legend(frameon=False, fontsize=4.8, loc='upper right', ncol=2)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax.set_axisbelow(True)

# Panel B: Improvement deltas (re-implementation)
ax2 = axes[1]
datasets = ['VOC\n(20 cls)', 'COCO\n(80 cls)']
xd = np.arange(len(datasets))
width = 0.22
d_corloc = [13.7, 11.4]
d_mask   = [18.0, 15.8]
d_pib    = [-3.0, -9.7]

bars1 = ax2.bar(xd - width, d_corloc, width, label='CorLoc', color=COLORS[0],
                edgecolor='white', linewidth=0.4)
bars2 = ax2.bar(xd,         d_mask,   width, label='Mask IoU', color=COLORS[1],
                edgecolor='white', linewidth=0.4)
bars3 = ax2.bar(xd + width, d_pib,    width, label='PiB', color=COLORS[2],
                edgecolor='white', linewidth=0.4)
ax2.axhline(0, color='#333333', linewidth=0.6)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.3 if h >= 0 else -0.3
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset,
                 f'{h:+.1f}', ha='center', va=va, fontsize=5.5, color='#333333')
ax2.set_xticks(xd)
ax2.set_xticklabels(datasets, fontsize=6.5)
ax2.set_ylabel('Δ vs LaSt-ViT (pp)')
ax2.set_title('Improvement Deltas', fontweight='bold', fontsize=7.5, pad=6)
ax2.set_ylim(-13, 25)
ax2.set_yticks(range(-10, 26, 5))   # -13 not labeled
ax2.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1,
           bbox_to_anchor=(-0.02, 1.03), labelspacing=0.25, handlelength=1.2)
ax2.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax2.set_axisbelow(True)

# Panel C: Scatter — Mask IoU improvement stability
ax3 = axes[2]
voc_miou_gain = 31.27 - 13.31
coco_miou_gain = 28.73 - 12.95
voc_cor_gain = 79.43 - 65.70
coco_cor_gain = 72.96 - 61.53

ax3.scatter([voc_cor_gain], [voc_miou_gain], s=80, color='#CC6677',
            edgecolor='white', linewidth=0.8, zorder=4)
ax3.annotate('VOC\n(20 classes)', (voc_cor_gain, voc_miou_gain),
             textcoords='offset points', xytext=(-10, 0), fontsize=5.5,
             color='#CC6677', fontweight='bold', ha='right', va='center')
ax3.scatter([coco_cor_gain], [coco_miou_gain], s=80, color='#88CCEE',
            edgecolor='white', linewidth=0.8, zorder=4)
ax3.annotate('COCO\n(80 classes)', (coco_cor_gain, coco_miou_gain),
             textcoords='offset points', xytext=(8, -5), fontsize=5.5,
             color='#88CCEE', fontweight='bold')

# Stability region
ax3.axhspan(15, 20, color='#CC6677', alpha=0.08)
ax3.axvspan(10, 15, color='#88CCEE', alpha=0.08)

ax3.set_xlabel('CorLoc Improvement (pp)')
ax3.set_ylabel('Mask IoU Improvement (pp)')
ax3.set_title('Gain Stability Across Datasets', fontweight='bold', fontsize=7.5, pad=6)
ax3.set_xlim(8, 20)
ax3.set_ylim(12, 24)
ax3.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax3.set_axisbelow(True)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#CC6677', alpha=0.15, label=' Mask IoU gain zone'),
    Patch(facecolor='#88CCEE', alpha=0.15, label=' CorLoc gain zone')
]
ax3.legend(handles=legend_elements, frameon=False, fontsize=6.5, loc='upper left',
           ncol=1, bbox_to_anchor=(0.0, 1.0), labelspacing=0.25, handlelength=1.5)

fig.suptitle('Cross-Dataset Consistency: DAFB-CLS vs LaSt-ViT (DINO ViT-S/16)',
             fontsize=8.5, fontweight='bold', y=1.02)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'cross_dataset', 'cross_dataset_all.pdf')
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
