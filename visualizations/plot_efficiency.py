"""
Nature-style figure: Efficiency analysis (params, latency, memory)
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

COLORS3 = ['#BBBBBB', '#88CCEE', '#CC6677']

# ── Data ────────────────────────────────────────────────────
methods = ['Baseline', 'LaSt-ViT', 'DAFB-CLS']

# DINO ViT-S/16
dino_params  = [21.67, 22.25, 22.25]  # M
dino_train   = [0.0004, 0.59, 0.59]   # M
dino_lat     = [6.03, 6.36, 7.71]     # ms
dino_mem     = [96.6, 184.2, 275.7]   # MB
dino_flops   = [4.25, 4.26, 4.30]     # G

# OpenCLIP ViT-B/16
clip_params  = [86.21, 87.94, 87.94]  # M
clip_train   = [0.016, 1.74, 1.74]    # M
clip_lat     = [13.60, 14.51, 16.08]  # ms
clip_mem     = [347.3, 686.5, 1032.5] # MB
clip_flops   = [2.91, 11.37, 11.42]   # G

# ── Figure 1: Latency comparison (both backbones) ──────────
fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.5))

# Panel A: DINO latency
ax = axes[0, 0]
x = np.arange(len(methods))
bars = ax.bar(x, dino_lat, 0.55, color=COLORS3, edgecolor='white', linewidth=0.4)
bars[-1].set_edgecolor('#333333')
bars[-1].set_linewidth(1.0)
for bar, val in zip(bars, dino_lat):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.2f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=6)
ax.set_ylabel('Latency (ms)')
ax.set_title('DINO ViT-S/16: Inference Latency', fontweight='bold', fontsize=7.5, pad=6)
ax.set_ylim(0, 20)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax.set_axisbelow(True)

# Panel B: OpenCLIP latency
ax2 = axes[0, 1]
bars2 = ax2.bar(x, clip_lat, 0.55, color=COLORS3, edgecolor='white', linewidth=0.4)
bars2[-1].set_edgecolor('#333333')
bars2[-1].set_linewidth(1.0)
for bar, val in zip(bars2, clip_lat):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}', ha='center', va='bottom', fontsize=6)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=6)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('OpenCLIP ViT-B/16: Inference Latency', fontweight='bold', fontsize=7.5, pad=6)
ax2.set_ylim(0, 20)
ax2.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax2.set_axisbelow(True)

# Panel C: Memory comparison (both)
ax3 = axes[1, 0]
x3 = np.arange(len(methods))
width = 0.3
bars_d = ax3.bar(x3 - width/2, dino_mem, width, label='DINO ViT-S/16',
                 color='#88CCEE', edgecolor='white', linewidth=0.4)
bars_c = ax3.bar(x3 + width/2, clip_mem, width, label='OpenCLIP ViT-B/16',
                 color='#CC6677', edgecolor='white', linewidth=0.4)
for bars, vals in [(bars_d, dino_mem), (bars_c, clip_mem)]:
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=5.5)
ax3.set_xticks(x3)
ax3.set_xticklabels(methods, fontsize=6)
ax3.set_ylabel('Peak GPU Memory (MB)')
ax3.set_title('Memory Usage', fontweight='bold', fontsize=7.5, pad=6)
ax3.legend(frameon=False, fontsize=5.5, loc='upper left')
ax3.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax3.set_axisbelow(True)

# Panel D: Trainable params pie-like horizontal bar
ax4 = axes[1, 1]
backbone_names = ['DINO ViT-S/16', 'OpenCLIP ViT-B/16']
trainable = [0.59, 1.74]  # M
backbone_p = [21.67, 86.21]
pct = [t / b * 100 for t, b in zip(trainable, backbone_p)]
y_pos = np.arange(len(backbone_names))
bh = ax4.barh(y_pos, pct, 0.5, color=['#88CCEE', '#CC6677'], edgecolor='white', linewidth=0.4)
for bar, val, tr, bp in zip(bh, pct, trainable, backbone_p):
    ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{tr:.2f}M / {bp:.1f}M\n({val:.1f}%)',
             va='center', fontsize=6, color='#333333')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(backbone_names, fontsize=6.5)
ax4.set_xlabel('Trainable Parameters (% of backbone)')
ax4.set_title('Parameter Efficiency', fontweight='bold', fontsize=7.5, pad=6)
ax4.set_xlim(0, 4.5)
ax4.xaxis.grid(True, linewidth=0.3, alpha=0.5)
ax4.set_axisbelow(True)
ax4.invert_yaxis()

fig.suptitle('Efficiency Analysis (RTX 4070 Laptop, batch=1, 224×224)',
             fontsize=8.5, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'efficiency', 'efficiency_all.pdf')
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

# ── Figure 2: Overhead multiplier comparison ───────────────
fig2, ax_o = plt.subplots(figsize=(3.5, 2.5))
backbone_labels = ['DINO\nViT-S/16', 'OpenCLIP\nViT-B/16']
lat_overhead = [7.71 / 6.03, 16.08 / 13.60]
mem_overhead = [275.7 / 96.6, 1032.5 / 347.3]
xo = np.arange(len(backbone_labels))
width = 0.3
bars_lo = ax_o.bar(xo - width/2, lat_overhead, width, label='Latency overhead (×)',
                   color='#CC6677', edgecolor='white', linewidth=0.4)
bars_mo = ax_o.bar(xo + width/2, mem_overhead, width, label='Memory overhead (×)',
                   color='#88CCEE', edgecolor='white', linewidth=0.4)
ax_o.axhline(1.0, color='#999999', linewidth=0.8, linestyle='--', label='No overhead')
for bars, vals in [(bars_lo, lat_overhead), (bars_mo, mem_overhead)]:
    for bar, val in zip(bars, vals):
        ax_o.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f'{val:.2f}×', ha='center', va='bottom', fontsize=6)
ax_o.set_xticks(xo)
ax_o.set_xticklabels(backbone_labels, fontsize=6.5)
ax_o.set_ylabel('Overhead Multiplier')
ax_o.set_title('DAFB-CLS Overhead vs Backbone Baseline', fontweight='bold', fontsize=8, pad=8)
ax_o.legend(frameon=False, fontsize=5.5, loc='upper left')
ax_o.set_ylim(0, 4.0)
ax_o.yaxis.grid(True, linewidth=0.3, alpha=0.5)
ax_o.set_axisbelow(True)
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), 'efficiency', 'overhead_multiplier.pdf')
fig2.savefig(out2, bbox_inches='tight')
fig2.savefig(out2.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()
