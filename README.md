# DAFB-CLS: Depth-Adaptive Foreground-Background CLS Decoupling for Vision Transformers

## Overview

DAFB-CLS is a post-hoc aggregator framework that decouples the CLS token in Vision Transformers into foreground and background semantic streams, with depth-wise adaptive attention for each stream independently.

### Core Problems Addressed

1. **Lazy Aggregation (LaSt-ViT)**: ViT CLS token tends to aggregate global semantics from background patches as shortcuts, due to coarse-grained semantic supervision and global attention.

2. **Lazy Accumulation (Attention Residuals)**: Standard residual connections perform fixed equal-weight accumulation of historical layer outputs, lacking selectivity along the depth dimension.

### Our Solution

DAFB-CLS unifies both problems into a single framework:

- **Multi-cue Foregroundness**: Beyond frequency stability alone, combining frequency, depth consistency, semantic alignment, and spatial compactness cues.
- **Adaptive Foreground Budget**: Replacing fixed Top-K with image-adaptive soft foreground mask prediction.
- **Dual CLS Decoupling**: Not suppressing background, but maintaining separate foreground/background CLS tokens with task-adaptive fusion.
- **Depth-wise Attention**: Foreground and background streams independently perform depth-wise softmax attention over historical block representations, inspired by Attention Residuals.

## Project Structure

```
dafb_cls/
├── models/
│   ├── feature_extractor.py   # MultiLayerFeatureExtractor
│   ├── cues.py                # Frequency, Depth, Semantic, Spatial cues
│   ├── foregroundness.py      # ForegroundnessHead
│   ├── adaptive_budget.py     # AdaptiveBudgetModule
│   ├── dual_cls.py            # DualCLSAggregator
│   ├── depth_attention.py     # ForegroundBackgroundDepthAttention
│   ├── fusion.py              # TaskAdaptiveFusionHead
│   └── heads.py               # Task heads (classification, segmentation, discovery)
├── losses/
│   ├── mask_losses.py         # Foreground mask losses (BCE + Dice)
│   ├── decouple_loss.py       # Foreground-background decoupling loss
│   └── budget_loss.py         # Budget regularization loss
├── datasets/
│   ├── voc.py                 # VOC dataset with bbox/mask support
│   └── coco.py                # COCO dataset loader
├── tools/
│   ├── train_posthoc.py       # Main training script
│   ├── extract_features.py    # Feature caching utility
│   ├── eval_pib.py            # Point-in-Box evaluation
│   ├── eval_mask_iou.py       # Mask IoU evaluation
│   ├── eval_corloc.py         # CorLoc for object discovery
│   ├── eval_ovseg.py          # Open-vocabulary segmentation
│   ├── eval_lastvit.py        # LaSt-ViT baseline evaluation
│   ├── run_ablation.py        # Ablation experiment runner
│   ├── stress_test.py         # Stress test across 4 subsets
│   ├── visualize_masks.py     # Mask visualization
│   └── visualize_depth_attention.py  # Depth attention visualization
└── configs/
    ├── openclip_vitb16_voc.yaml
    ├── dino_vits16_voc.yaml
    ├── ablation.yaml          # OpenCLIP ablation variants
    ├── dino_ablation.yaml     # DINO ablation variants
    └── dino_improved.yaml     # Stable background improvement config
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision open_clip_torch timm pyyaml Pillow numpy tqdm
```

### Training (OpenCLIP ViT-B/16 + VOC)

```bash
python -m dafb_cls.tools.train_posthoc --config dafb_cls/configs/openclip_vitb16_voc.yaml
```

### Training (DINO ViT-S/16 + VOC)

```bash
python -m dafb_cls.tools.train_posthoc --config dafb_cls/configs/dino_vits16_voc.yaml
```

### Evaluation

```bash
python -m dafb_cls.tools.eval_pib --checkpoint <ckpt> --config <cfg>
python -m dafb_cls.tools.eval_mask_iou --checkpoint <ckpt> --config <cfg>
python -m dafb_cls.tools.eval_corloc --checkpoint <ckpt> --config <cfg>
python -m dafb_cls.tools.eval_ovseg --checkpoint <ckpt> --config <cfg>
```

### Visualization

```bash
python -m dafb_cls.tools.visualize_masks --checkpoint <ckpt> --config <cfg>
python -m dafb_cls.tools.visualize_depth_attention --checkpoint <ckpt> --config <cfg>
```

## Backbone Support

| Backbone | Use Case | Default Layers |
|----------|----------|---------------|
| OpenCLIP ViT-B/16 | Open-vocabulary segmentation | [3, 6, 9, 12] |
| DINO ViT-S/16 | Object discovery | [3, 6, 9, 12] |
| Supervised ViT/DeiT | Classification / PiB | [3, 6, 9, 12] |

## Method Summary

**Input**: Image x -> Frozen ViT Backbone -> Multi-layer patch features {x^l}_{l in L}

**Step 1 - Multi-cue Foregroundness**:
- S_i: Frequency stability (LaSt-ViT style FFT low-pass)
- D_i: Depth consistency (cross-layer activation persistence)
- A_i: Semantic alignment (patch-CLS or patch-text similarity)
- C_i: Spatial compactness (local neighborhood smoothing)
- F_i = w_s*S_i + w_d*D_i + w_a*A_i + w_c*C_i

**Step 2 - Adaptive Budget**:
- tau predicted from global image feature
- m_i^F = sigmoid((F_i - tau) / T)

**Step 3 - Dual CLS**:
- B_l^F = sum_i m_i^F * x_i^l / sum_i m_i^F
- B_l^B = sum_i m_i^B * x_i^l / sum_i m_i^B

**Step 4 - Depth Attention**:
- beta_l^F = softmax(q_F^T * LN(B_l^F))
- beta_l^B = softmax(q_B^T * LN(B_l^B))
- C_F = sum_l beta_l^F * B_l^F
- C_B = sum_l beta_l^B * B_l^B

**Step 5 - Task-Adaptive Fusion**:
- g = sigmoid(MLP([C_F, C_B, global_feat]))
- C = g * C_F + (1-g) * C_B

## Experiment Results (updated 2026-05-16)

### DINO ViT-S/16 + VOC (Object Discovery)

| Method | CorLoc (%) | Mask IoU (%) | PiB (%) |
|--------|:----------:|:------------:|:-------:|
| ViT baseline (CLS only) | 29.33 | 30.76 | 42.72 |
| LaSt-ViT (FFT stability, K=59) | 65.70 | 13.31 | 48.10 |
| **DAFB-CLS (full)** | **79.43** | **31.27** | 45.07 |

### OpenCLIP ViT-B/16 + VOC (Segmentation)

| Metric | Score |
|--------|-------|
| Best mIoU | **61.70%** |
| CorLoc | 77.29% |
| Mask IoU | 33.32% |
| PiB | 79.64% |

### DINO ViT-S/16 + COCO (Object Discovery)

| Method | CorLoc (%) | Mask IoU (%) | PiB (%) |
|--------|:----------:|:------------:|:-------:|
| LaSt-ViT (FFT stability, K=147) | 61.53 | 12.95 | 45.09 |
| **DAFB-CLS (full)** | **72.96** | **28.73** | 35.44 |

### Cross-Dataset Consistency

| Improvement | VOC (20cls) | COCO (80cls) |
|-------------|:-----------:|:------------:|
| CorLoc | +13.7pp | +11.4pp |
| Mask IoU | +18.0pp | +15.8pp |

### Comparison Methods (DINO ViT-S/16)

**VOC:**

| Method | CorLoc | Mask IoU | PiB |
|--------|:------:|:--------:|:---:|
| CAM | 62.80% | 3.52% | 21.26% |
| DINO-seg | 68.46% | 8.83% | 24.64% |
| LaSt-ViT | 65.70% | 13.31% | 48.10% |
| **DAFB-CLS** | **79.43%** | **31.27%** | 45.07% |

**COCO:**

| Method | CorLoc | Mask IoU | PiB |
|--------|:------:|:--------:|:---:|
| CAM | 59.81% | 4.30% | 24.96% |
| DINO-seg | 67.23% | 9.24% | 29.00% |
| LaSt-ViT | 61.53% | 12.95% | 45.09% |
| **DAFB-CLS** | **72.96%** | **28.73%** | 35.44% |

### Ablation (DINO ViT-S/16)

| Variant | CorLoc% | MaskIoU% | PiB% |
|---------|:-------:|:--------:|:----:|
| baseline_dino | 29.33 | 30.76 | 42.72 |
| full | **79.43** | **31.27** | 45.07 |
| no_budget | 79.78 | 13.91 | 53.62 |
| no_cues | 47.48 | 30.16 | 31.68 |
| no_dual_cls | 78.40 | 31.25 | 33.13 |
| shared_depth | 77.43 | 31.18 | 28.36 |
| hard_topk | 81.71 | 22.68 | 51.76 |

### Stress Test (DINO full, 4 subsets)

| Subset | Samples | FG IoU | PiB | Pearson r |
|--------|---------|--------|-----|-----------|
| stable_background | 551 | 13.02% | 20.87% | 0.394 |
| textured_foreground | 898 | 34.42% | 43.88% | 0.635 |
| small_object | 450 | 26.92% | 34.67% | 0.645 |
| multi_object | 436 | 36.19% | 43.12% | 0.605 |

### Key Findings

1. DAFB-CLS improves CorLoc by **+50pp** over raw ViT and **+13.7pp** over LaSt-ViT on VOC
2. Consistent on COCO: CorLoc **+11.4pp**, Mask IoU **+15.8pp** over LaSt-ViT
3. Multi-cue foregroundness is essential (no_cues: CorLoc drops to 47%)
4. Adaptive budget prevents masking collapse (no_budget: Mask IoU drops to 14%)
5. Dual CLS critical for spatial localization (no_dual_cls: PiB drops to 33%)
6. Stable background (sky/walls) is the main failure mode (FG IoU 13%)
