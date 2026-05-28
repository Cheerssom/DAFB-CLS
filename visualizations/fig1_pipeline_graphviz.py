"""
DAFB-CLS Pipeline Architecture — Graphviz Version
Auto-layout directed graph with publication styling.
"""
import subprocess
import os

DOT_SOURCE = r'''
digraph DAFBCLS {
    // ── Global Settings ─────────────────────────────
    rankdir=TB;
    splines=ortho;
    nodesep=0.35;
    ranksep=0.5;
    bgcolor="#FAFAFA";
    fontname="Arial";
    dpi=300;
    size="3.5,7!";
    ratio=compress;

    // ── Node Style Defaults ─────────────────────────
    node [
        fontname="Arial",
        fontsize=10,
        style="rounded,filled",
        penwidth=0.8,
        margin="0.15,0.08"
    ];

    edge [
        fontname="Arial",
        fontsize=8,
        color="#666666",
        penwidth=0.8,
        arrowsize=0.7
    ];

    // ── (a) Input & Backbone ────────────────────────
    input [
        label="Input Image x",
        shape=box,
        fillcolor="#E8E8E8",
        fontcolor="#1A1A1A",
        color="#CCCCCC"
    ];

    backbone [
        label="Frozen ViT Backbone Φ\n(DINO / OpenCLIP)",
        shape=box,
        fillcolor="#4A6FA5",
        fontcolor="white",
        color="#3A5A8A"
    ];

    features [
        label="Multi-Layer Feature Extraction\n{P^l}_{l∈ℒ},  c_cls",
        shape=box,
        fillcolor="#5A7A9A",
        fontcolor="white",
        color="#4A6A8A"
    ];

    input -> backbone -> features;

    // ── (b) Four Cues ───────────────────────────────
    subgraph cluster_cues {
        label="Spatial Selection";
        labeljust=l;
        style="dashed,rounded";
        color="#BBBBBB";
        fontsize=9;
        fontcolor="#888888";
        penwidth=0.6;

        freq  [label="Frequency\nStability  S_i", fillcolor="#6B8E6B", fontcolor="white", color="#5A7A5A"];
        depth [label="Depth\nConsistency  D_i", fillcolor="#8FA86E", fontcolor="white", color="#7A945A"];
        sem   [label="Semantic\nAlignment  A_i", fillcolor="#A89B6E", fontcolor="white", color="#94885A"];
        spat  [label="Spatial\nCompactness  C_i", fillcolor="#6E8FA8", fontcolor="white", color="#5A7A94"];

        {rank=same; freq; depth; sem; spat;}

        freq -> fg_score [style=invis];
        depth -> fg_score [style=invis];
        sem -> fg_score [style=invis];
        spat -> fg_score [style=invis];
    }

    features -> freq;
    features -> depth;
    features -> sem;
    features -> spat;

    fg_score [
        label="Foregroundness Scoring\nF_i = Σ w_k·s_{i,k} + MLP(s_i) + Smooth",
        shape=box,
        fillcolor="#8B6B4A",
        fontcolor="white",
        color="#7A5A3A"
    ];

    mask [
        label="Adaptive Budget Masking\nm_i^F = σ((F_i − τ) / T)",
        shape=box,
        fillcolor="#C47A5A",
        fontcolor="white",
        color="#A86A4A"
    ];

    {rank=same; fg_score; mask;}
    fg_score -> mask;

    // ── (c) Dual CLS ────────────────────────────────
    subgraph cluster_dual {
        label="Dual CLS with Depth Attention";
        labeljust=c;
        style="dashed,rounded";
        color="#BBBBBB";
        fontsize=9;
        fontcolor="#888888";
        penwidth=0.6;

        subgraph cluster_fg {
            label="";
            style="invis";

            cls_fg [
                label="Foreground CLS\nB_l^F = Σ m_i^F · P_i^l / Σ m_i^F",
                shape=box,
                fillcolor="#D4845A",
                fontcolor="white",
                color="#B87040"
            ];
            da_fg [
                label="Depth Attention\nC_F = Σ β_l^F · B_l^F",
                shape=box,
                fillcolor="#9B7DB8",
                fontcolor="white",
                color="#8868A0"
            ];
            cls_fg -> da_fg;
        }

        subgraph cluster_bg {
            label="";
            style="invis";

            cls_bg [
                label="Background CLS\nB_l^B = Σ m_i^B · P_i^l / Σ m_i^B",
                shape=box,
                fillcolor="#5A8AB4",
                fontcolor="white",
                color="#4070A0"
            ];
            da_bg [
                label="Depth Attention\nC_B = Σ β_l^B · B_l^B",
                shape=box,
                fillcolor="#9B7DB8",
                fontcolor="white",
                color="#8868A0"
            ];
            cls_bg -> da_bg;
        }
    }

    mask -> cls_fg;
    mask -> cls_bg;

    // ── (d) Fusion ──────────────────────────────────
    fusion [
        label="Task-Adaptive Fusion\ng = σ(MLP([C_F; C_B; c_cls]))\nC = g·C_F + (1−g)·C_B",
        shape=box,
        fillcolor="#C75D5D",
        fontcolor="white",
        color="#A84A4A"
    ];

    da_fg -> fusion;
    da_bg -> fusion;

    // ── (e) Task Heads ──────────────────────────────
    head_cls  [label="Classification\nMLP(C) → y",  shape=box, fillcolor="#6B6B6B", fontcolor="white", color="#555555"];
    head_seg  [label="Segmentation\nSim(P, t_j) → M", shape=box, fillcolor="#6B6B6B", fontcolor="white", color="#555555"];
    head_disc [label="Object Discovery\nScore(P) → S", shape=box, fillcolor="#6B6B6B", fontcolor="white", color="#555555"];

    {rank=same; head_cls; head_seg; head_disc;}

    fusion -> head_cls;
    fusion -> head_seg;
    fusion -> head_disc;
}
'''

out_dir = 'E:/DAFB-CLS/figures'
os.makedirs(out_dir, exist_ok=True)

dot_path = f'{out_dir}/fig1_pipeline_graphviz.dot'
with open(dot_path, 'w', encoding='utf-8') as f:
    f.write(DOT_SOURCE)

# Try rendering with graphviz
formats = ['pdf', 'svg', 'png']
for fmt in formats:
    out_file = f'{out_dir}/fig1_pipeline_graphviz.{fmt}'
    try:
        result = subprocess.run(
            ['dot', f'-T{fmt}', dot_path, '-o', out_file],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f'✅ Saved: {out_file}')
        else:
            print(f'❌ Error rendering {fmt}: {result.stderr}')
    except FileNotFoundError:
        print('❌ Graphviz not found. Install: winget install Graphviz.Graphviz')
        print(f'   Or render manually: dot -T{fmt} {dot_path} -o {out_file}')
        break
    except Exception as e:
        print(f'❌ Error: {e}')
        break

print(f'\nDOT source saved to: {dot_path}')
