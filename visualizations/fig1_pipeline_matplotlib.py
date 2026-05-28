"""
DAFB-CLS Pipeline Architecture — Publication-Quality Figure (Matplotlib)
IEEE two-column format: ~3.5" × 6" (89mm × 152mm)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Publication Style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

# ── Color Palette (low-saturation, Nature-style) ───────────────
C = {
    'backbone':  '#4A6FA5',   # steel blue
    'cue_freq':  '#6B8E6B',   # sage green
    'cue_depth': '#8FA86E',   # olive
    'cue_sem':   '#A89B6E',   # sand
    'cue_spat':  '#6E8FA8',   # slate
    'mask':      '#C47A5A',   # terracotta
    'fg':        '#D4845A',   # warm orange
    'bg':        '#5A8AB4',   # cool blue
    'depth':     '#9B7DB8',   # lavender
    'fusion':    '#C75D5D',   # muted red
    'head':      '#6B6B6B',   # charcoal
    'arrow':     '#555555',
    'text':      '#1A1A1A',
    'bg_canvas': '#FAFAFA',
    'bg_panel':  '#FFFFFF',
    'highlight': '#FFF3E0',
}

# ── Figure Setup ───────────────────────────────────────────────
fig = plt.figure(figsize=(3.5, 6.5), dpi=300)
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C['bg_canvas'])

# ── Helper Functions ───────────────────────────────────────────
def rounded_box(x, y, w, h, color, label='', fontsize=6, textcolor='white',
                radius=0.15, alpha=0.92, edgecolor=None, linewidth=0.6,
                label2='', fontsize2=5.5):
    """Draw a rounded rectangle with centered label."""
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=ec,
                         linewidth=linewidth, alpha=alpha,
                         zorder=3)
    ax.add_patch(box)
    if label:
        ly = y + h/2 + (0.08 if label2 else 0)
        ax.text(x + w/2, ly, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)
    if label2:
        ax.text(x + w/2, y + h/2 - 0.15, label2, ha='center', va='center',
                fontsize=fontsize2, color=textcolor, alpha=0.85, zorder=4)
    return box

def arrow(x1, y1, x2, y2, color=None, lw=0.9, style='->', shrink=0, zorder=2):
    """Draw an arrow between two points."""
    c = color or C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                               shrinkA=shrink, shrinkB=shrink),
                zorder=zorder)

def label_text(x, y, text, fontsize=5.5, color=None, ha='center', style='italic'):
    """Add a small annotation label."""
    c = color or C['text']
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            color=c, fontstyle=style, zorder=4)

def section_bracket(x, y_top, y_bot, label, color, side='left'):
    """Draw a vertical bracket with section label."""
    mid = (y_top + y_bot) / 2
    lw = 0.5
    if side == 'left':
        ax.plot([x, x], [y_bot, y_top], color=color, lw=lw, zorder=2)
        ax.plot([x, x+0.15], [y_bot, y_bot], color=color, lw=lw, zorder=2)
        ax.plot([x, x+0.15], [y_top, y_top], color=color, lw=lw, zorder=2)
        ax.text(x-0.15, mid, label, ha='right', va='center', fontsize=5,
                color=color, fontweight='bold', rotation=90, zorder=4)
    else:
        ax.plot([x, x], [y_bot, y_top], color=color, lw=lw, zorder=2)
        ax.plot([x-0.15, x], [y_bot, y_bot], color=color, lw=lw, zorder=2)
        ax.plot([x-0.15, x], [y_top, y_top], color=color, lw=lw, zorder=2)
        ax.text(x+0.15, mid, label, ha='left', va='center', fontsize=5,
                color=color, fontweight='bold', rotation=90, zorder=4)


# ══════════════════════════════════════════════════════════════
# LAYOUT (top to bottom, y: 17.5 → 0.5)
# ══════════════════════════════════════════════════════════════

cx = 5.0  # center x
W = 7.0   # main width

# ── (a) Input + Frozen Backbone ────────────────────────────────
y_img = 17.0
rounded_box(cx-1.2, y_img, 2.4, 0.6, '#E8E8E8', 'Input Image $x$',
            fontsize=6.5, textcolor=C['text'], edgecolor='#CCCCCC')

y_bb = 15.8
rounded_box(cx-W/2, y_bb, W, 0.7, C['backbone'],
            'Frozen ViT Backbone  $\\Phi$',
            fontsize=7, label2='(DINO / OpenCLIP)')

arrow(cx, y_img, cx, y_bb+0.7, lw=1.0)

# ── (b) Multi-Layer Feature Extraction ─────────────────────────
y_feat = 14.6
rounded_box(cx-W/2, y_feat, W, 0.7, '#5A7A9A',
            'Multi-Layer Feature Extraction',
            fontsize=6.5, label2='$\\{P^l\\}_{l \\in \\mathcal{L}}$,  $c_{cls}$')

arrow(cx, y_bb, cx, y_feat+0.7, lw=0.9)

# ── (c) Four Cues (horizontal) ─────────────────────────────────
y_cue = 13.2
cue_w = 1.45
cue_h = 0.65
cue_gap = 0.2
total_cue_w = 4 * cue_w + 3 * cue_gap
cue_start = cx - total_cue_w / 2

cues = [
    ('Freq.\nStability', C['cue_freq'], '$S_i$'),
    ('Depth\nConsistency', C['cue_depth'], '$D_i$'),
    ('Semantic\nAlignment', C['cue_sem'], '$A_i$'),
    ('Spatial\nCompactness', C['cue_spat'], '$C_i$'),
]

cue_centers = []
for i, (name, color, sym) in enumerate(cues):
    x = cue_start + i * (cue_w + cue_gap)
    rounded_box(x, y_cue, cue_w, cue_h, color, name, fontsize=5.5)
    label_text(x + cue_w/2, y_cue - 0.15, sym, fontsize=5.5, color=color)
    cue_centers.append(x + cue_w/2)

# Arrows from features to cues
for xc in cue_centers:
    arrow(xc, y_feat, xc, y_cue + cue_h, lw=0.7)

# ── (d) Foregroundness Scoring ─────────────────────────────────
y_fg = 11.8
rounded_box(cx-W/2, y_fg, W, 0.75, '#8B6B4A',
            'Foregroundness Scoring',
            fontsize=6.5,
            label2='$F_i = \\sum w_k s_{i,k} + \\mathrm{MLP}(\\mathbf{s}_i) + 0.5 \\cdot \\mathrm{Smooth}$')

# Arrows from cues down
for xc in cue_centers:
    arrow(xc, y_cue, xc, y_fg + 0.75, lw=0.6)
    # converge to center
    mid_y = (y_cue + y_fg + 0.75) / 2
    ax.plot([xc, cx], [mid_y, mid_y], color=C['arrow'], lw=0.4, zorder=2)

# ── (e) Adaptive Masking ───────────────────────────────────────
y_mask = 10.5
rounded_box(cx-2.5, y_mask, 5.0, 0.7, C['mask'],
            'Adaptive Budget Masking',
            fontsize=6.5,
            label2='$m_i^F = \\sigma((F_i - \\tau) / T)$')

arrow(cx, y_fg, cx, y_mask + 0.7, lw=0.9)

# ── Dashed box around (c)+(d)+(e) ──────────────────────────────
dashed_box = FancyBboxPatch(
    (cx - W/2 - 0.2, y_cue - 0.3), W + 0.4, (y_mask + 0.7) - (y_cue - 0.3) + 0.15,
    boxstyle="round,pad=0.1", facecolor='none', edgecolor='#BBBBBB',
    linewidth=0.5, linestyle='--', zorder=1)
ax.add_patch(dashed_box)
ax.text(cx - W/2 - 0.05, y_mask + 0.85, 'Spatial Selection',
        fontsize=5.5, color='#888888', fontweight='bold', zorder=4)

# ── (f) Dual CLS Aggregation ───────────────────────────────────
y_dual = 8.9
dual_w = 3.0
dual_h = 0.8

rounded_box(cx - dual_w - 0.25, y_dual, dual_w, dual_h, C['fg'],
            'Foreground CLS', fontsize=6.5,
            label2='$B_l^F = \\frac{\\sum m_i^F \\cdot P_i^l}{\\sum m_i^F}$')

rounded_box(cx + 0.25, y_dual, dual_w, dual_h, C['bg'],
            'Background CLS', fontsize=6.5,
            label2='$B_l^B = \\frac{\\sum m_i^B \\cdot P_i^l}{\\sum m_i^B}$')

# Arrows from mask to dual
arrow(cx - 1.5, y_mask, cx - dual_w/2 - 0.25, y_dual + dual_h, lw=0.8)
arrow(cx + 1.5, y_mask, cx + dual_w/2 + 0.25, y_dual + dual_h, lw=0.8)

# ── (g) Depth Attention ────────────────────────────────────────
y_depth = 7.3
rounded_box(cx - dual_w - 0.25, y_depth, dual_w, 0.8, C['depth'],
            'Depth Attention', fontsize=6,
            label2='$C_F = \\sum \\beta_l^F \\cdot B_l^F$')

rounded_box(cx + 0.25, y_depth, dual_w, 0.8, C['depth'],
            'Depth Attention', fontsize=6,
            label2='$C_B = \\sum \\beta_l^B \\cdot B_l^B$')

# Arrows
arrow(cx - dual_w/2 - 0.25, y_dual, cx - dual_w/2 - 0.25, y_depth + 0.8, lw=0.8)
arrow(cx + dual_w/2 + 0.25, y_dual, cx + dual_w/2 + 0.25, y_depth + 0.8, lw=0.8)

# ── Dashed box around (f)+(g) ──────────────────────────────────
dashed_box2 = FancyBboxPatch(
    (cx - dual_w - 0.5, y_depth - 0.15), 2 * dual_w + 1.0,
    (y_dual + dual_h) - (y_depth - 0.15) + 0.2,
    boxstyle="round,pad=0.1", facecolor='none', edgecolor='#BBBBBB',
    linewidth=0.5, linestyle='--', zorder=1)
ax.add_patch(dashed_box2)
ax.text(cx, y_dual + dual_h + 0.3, 'Dual CLS with Depth Attention',
        fontsize=5.5, color='#888888', fontweight='bold', ha='center', zorder=4)

# ── (h) Task-Adaptive Fusion ───────────────────────────────────
y_fusion = 5.7
rounded_box(cx - 2.2, y_fusion, 4.4, 0.8, C['fusion'],
            'Task-Adaptive Fusion', fontsize=6.5,
            label2='$g = \\sigma(\\mathrm{MLP}([C_F; C_B; c_{cls}]))$')

arrow(cx - dual_w/2 - 0.25, y_depth, cx - 1.0, y_fusion + 0.8, lw=0.8)
arrow(cx + dual_w/2 + 0.25, y_depth, cx + 1.0, y_fusion + 0.8, lw=0.8)

# Gate formula
label_text(cx, y_fusion - 0.25, '$C = g \\cdot C_F + (1-g) \\cdot C_B$',
           fontsize=6, color=C['fusion'])

# ── (i) Task Heads ─────────────────────────────────────────────
y_head = 3.8
head_w = 2.0
head_gap = 0.5
total_head = 3 * head_w + 2 * head_gap
head_start = cx - total_head / 2

heads = [
    ('Classification', '$\\mathrm{MLP}(C) \\to y$'),
    ('Segmentation', '$\\mathrm{Sim}(P, t_j) \\to M$'),
    ('Object Discovery', '$\\mathrm{Score}(P) \\to S$'),
]

head_centers = []
for i, (name, formula) in enumerate(heads):
    x = head_start + i * (head_w + head_gap)
    rounded_box(x, y_head, head_w, 0.7, C['head'], name, fontsize=5.5,
                label2=formula, fontsize2=5)
    head_centers.append(x + head_w/2)

# Arrows from fusion to heads
for hc in head_centers:
    arrow(cx, y_fusion, hc, y_head + 0.7, lw=0.7)

# ── Section Labels (right bracket) ─────────────────────────────
section_bracket(9.2, y_bb + 0.7, y_feat, '(a)', C['backbone'], side='right')
section_bracket(9.2, y_cue + cue_h, y_mask, '(b)', '#8B6B4A', side='right')
section_bracket(9.2, y_dual + dual_h, y_depth, '(c)', C['depth'], side='right')
section_bracket(9.2, y_fusion + 0.8, y_fusion, '(d)', C['fusion'], side='right')
section_bracket(9.2, y_head + 0.7, y_head, '(e)', C['head'], side='right')

# ── Key formulas box (bottom) ──────────────────────────────────
y_key = 1.8
key_box = FancyBboxPatch(
    (cx - W/2, y_key - 0.1), W, 1.2,
    boxstyle="round,pad=0.15", facecolor='#F5F5F5', edgecolor='#DDDDDD',
    linewidth=0.4, zorder=1)
ax.add_patch(key_box)

formulas = [
    ('Cues', '$\\mathbf{s}_i = [S_i, D_i, A_i, C_i]$'),
    ('Mask', '$m_i^F = \\sigma((F_i - \\tau) / T)$'),
    ('Depth', '$\\beta_l = \\mathrm{softmax}(w^T \\cdot \\mathrm{RMSNorm}(B_l))$'),
    ('Fusion', '$g = \\sigma(\\mathrm{MLP}([C_F; C_B; c_{cls}]))$'),
]

ax.text(cx - W/2 + 0.3, y_key + 0.9, 'Key Equations:', fontsize=5.5,
        fontweight='bold', color=C['text'], zorder=4)
for i, (name, formula) in enumerate(formulas):
    ax.text(cx - W/2 + 0.3, y_key + 0.55 - i * 0.28, f'{name}:  {formula}',
            fontsize=5.2, color='#444444', zorder=4)

# ── Title ──────────────────────────────────────────────────────
ax.text(cx, 17.8, 'DAFB-CLS Framework', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C['text'], zorder=4)

# ── Save ───────────────────────────────────────────────────────
out_dir = 'E:/DAFB-CLS/figures'
os.makedirs(out_dir, exist_ok=True)

fig.savefig(f'{out_dir}/fig1_pipeline_matplotlib.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{out_dir}/fig1_pipeline_matplotlib.svg', bbox_inches='tight')
fig.savefig(f'{out_dir}/fig1_pipeline_matplotlib.png', bbox_inches='tight', dpi=600)
plt.close()
print(f'Saved to {out_dir}/fig1_pipeline_matplotlib.{{pdf,svg,png}}')
