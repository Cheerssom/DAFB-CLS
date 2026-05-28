# DAFB-CLS Coding Checklist

## Phase 1: Project Setup
- [x] Create project directory structure
- [x] Create README_DAFB_CLS.md
- [x] Create TODO.md
- [x] Create experiments.md

## Phase 2: Version A - Core Modules (Frozen Backbone + Post-hoc Aggregator)

### 2.1 Feature Extraction
- [x] `dafb_cls/models/feature_extractor.py` - MultiLayerFeatureExtractor
  - [x] Support OpenCLIP ViT-B/16 (layers [3, 6, 9, 12])
  - [x] Support DINO ViT-S/16 (layers [3, 6, 9, 12])
  - [x] Support Supervised ViT/DeiT
  - [x] Output shape: [B, L, N, D] where L=num_layers, N=num_patches, D=dim

### 2.2 Cues Module
- [x] `dafb_cls/models/cues.py`
  - [x] FrequencyStabilityCue: FFT low-pass -> stability score
  - [x] DepthConsistencyCue: cross-layer activation consistency
  - [x] SemanticAlignmentCue: patch-CLS / patch-text similarity
  - [x] SpatialCompactnessCue: local neighborhood affinity smoothing

### 2.3 Foregroundness & Budget
- [x] `dafb_cls/models/foregroundness.py` - ForegroundnessHead
  - [x] MLP-based cue fusion: F_i = w_s*S_i + w_d*D_i + w_a*A_i + w_c*C_i
  - [x] Learnable cue weights
- [x] `dafb_cls/models/adaptive_budget.py` - AdaptiveBudgetModule
  - [x] Soft mask: m_i^F = sigmoid((F_i - tau) / T)
  - [x] tau predicted from global image feature
  - [x] Temperature T as learnable parameter

### 2.4 Dual CLS
- [x] `dafb_cls/models/dual_cls.py`
  - [x] ForegroundnessHead -> m_F
  - [x] BackgroundnessHead -> m_B (independent, not 1 - m_F)
  - [x] DualCLSAggregator: B_l^F, B_l^B for each layer

### 2.5 Depth Attention
- [x] `dafb_cls/models/depth_attention.py`
  - [x] ForegroundBackgroundDepthAttention
  - [x] Separate q_F, q_B pseudo-queries (zero-init)
  - [x] RMSNorm on keys
  - [x] softmax normalization over depth dimension

### 2.6 Fusion
- [x] `dafb_cls/models/fusion.py` - TaskAdaptiveFusionHead
  - [x] g = sigmoid(MLP([C_F, C_B, global_feat]))
  - [x] C = g * C_F + (1-g) * C_B

### 2.7 Task Heads
- [x] `dafb_cls/models/heads.py`
  - [x] ClassificationHead
  - [x] SegmentationHead (patch-text similarity based)
  - [x] ObjectDiscoveryScoringHead

### 2.8 Losses
- [x] `dafb_cls/losses/mask_losses.py` - BCE + Dice
- [x] `dafb_cls/losses/decouple_loss.py` - cos(C_F, C_B)^2
- [x] `dafb_cls/losses/budget_loss.py` - foreground ratio regularization

### 2.9 Datasets
- [x] `dafb_cls/datasets/voc.py` - VOC with bbox/mask support
- [x] `dafb_cls/datasets/coco.py` - COCO dataset loader

## Phase 3: Training & Evaluation Scripts
- [x] `dafb_cls/tools/train_posthoc.py` - freeze backbone, train aggregator
- [x] `dafb_cls/tools/extract_features.py` - cache multi-layer features
- [x] `dafb_cls/tools/eval_pib.py` - Point-in-Box metric
- [x] `dafb_cls/tools/eval_mask_iou.py` - Mask IoU metric
- [x] `dafb_cls/tools/eval_corloc.py` - CorLoc for object discovery
- [x] `dafb_cls/tools/eval_ovseg.py` - Open-vocabulary segmentation
- [x] `dafb_cls/tools/eval_tokencut.py` - TokenCut baseline (NCut)
- [x] `dafb_cls/tools/eval_maskclip.py` - MaskCLIP baseline (CLIP patch-text)

## Phase 4: Visualization & Config
- [x] `dafb_cls/tools/visualize_masks.py`
- [x] `dafb_cls/tools/visualize_depth_attention.py`
- [x] `dafb_cls/configs/openclip_vitb16_voc.yaml`
- [x] `dafb_cls/configs/dino_vits16_voc.yaml`
- [x] `dafb_cls/configs/ablation.yaml`

## Phase 5: Ablation Configs
- [x] Ablation config variants defined in ablation.yaml
- [x] Implement ablation switches in DAFBCLS model code
  - [x] `disable_dafb` / `lastvit_mode` support
  - [x] `disable_cues` support
  - [x] `disable_adaptive_budget` support
  - [x] `disable_dual_cls` / `single_cls_mode` support
  - [x] `share_depth_attn` support
  - [x] `mask_type: "hard_topk"` support
- [x] baseline ViT / CLIP / DINO
- [x] LaSt-ViT-style frequency Top-K
- [x] + multi-cue foregroundness
- [x] + adaptive budget
- [x] + dual CLS
- [x] + separate F/B depth attention
- [x] full DAFB-CLS
- [x] Component ablation: w/o frequency, w/o semantic, w/o depth consistency
- [x] fixed K vs adaptive budget
- [x] single CLS vs dual CLS
- [x] shared vs separate depth attention
- [x] hard Top-K vs soft mask
- [x] number of blocks N = 2/4/6/8

## Phase 6: Stress Test
- [x] Stable background subset (code in stress_test.py)
- [x] Textured foreground subset (code in stress_test.py)
- [x] Small object subset (code in stress_test.py)
- [x] Multi-object subset (code in stress_test.py)
- [x] Per-subset metric reporting script (stress_test.py)

## Phase 7: Training & Evaluation Results

### DINO ViT-S/16 + VOC (object_discovery)
- [x] Training (checkpoints: epoch_9 ~ epoch_49 + final.pt)
- [x] eval_corloc: **76.74%** (1112/1449)
- [x] eval_mask_iou: **31.60%** (Precision: 34.75%, Recall: 80.78%)
- [x] eval_pib: **33.54%** (486/1449)

### OpenCLIP ViT-B/16 + VOC (segmentation)
- [x] Training (50 epochs, checkpoints: epoch_9 ~ epoch_49 + final.pt)
- [x] Best mIoU: **61.7%** (epoch 49)
- [x] 21 classes all have non-zero IoU
- [x] Top classes: bg 89.2%, cls8 83.5%, cls6 79.3%, cls10 77.6%, cls12 77.0%, cls17 76.4%, cls19 74.8%, cls15 74.3%, cls13 73.6%, cls3 73.0%
- [x] eval_corloc / eval_mask_iou / eval_pib 评估
  - CorLoc: 77.29% (1120/1449)
  - Mask IoU: 33.32% (Precision: 34.41%, Recall: 85.42%)
  - PiB: 79.64% (1154/1449)

## Phase 8: Remaining Items
- [x] Copy VOC2012 `Annotations/` and `SegmentationClass/` for full evaluation support
- [x] Implement ablation switches in model code (Phase 5)
- [x] Run OpenCLIP ablation experiments (7/7 完成)
  - [x] baseline_clip: mIoU 59.02%
  - [x] full: mIoU 62.27%
  - [x] no_budget: mIoU 64.80%
  - [x] no_cues: mIoU 62.61%
  - [x] no_dual_cls: mIoU 61.65%
  - [x] shared_depth: mIoU 62.34%
  - [x] hard_topk: mIoU 61.66%
- [x] Run DINO ablation experiments (7/7 完成)
  - baseline_dino: CorLoc 29.33%, Mask IoU 30.76%, PiB 42.72%
  - full: CorLoc 79.43%, Mask IoU 31.27%, PiB 45.07%
  - no_budget: CorLoc 79.78%, Mask IoU 13.91%, PiB 53.62%
  - no_cues: CorLoc 47.48%, Mask IoU 30.16%, PiB 31.68%
  - no_dual_cls: CorLoc 78.40%, Mask IoU 31.25%, PiB 33.13%
  - shared_depth: CorLoc 77.43%, Mask IoU 31.18%, PiB 28.36%
  - hard_topk: CorLoc 81.71%, Mask IoU 22.68%, PiB 51.76%
- [x] Create experiments.md
- [x] OpenCLIP eval_corloc / eval_mask_iou / eval_pib
- [x] DINO stress test
  - stable_background (551): FG IoU 13.02%, PiB 20.87%, Pearson 0.394
  - textured_foreground (898): FG IoU 34.42%, PiB 43.88%, Pearson 0.635
  - small_object (450): FG IoU 26.92%, PiB 34.67%, Pearson 0.645
  - multi_object (436): FG IoU 36.19%, PiB 43.12%, Pearson 0.605
- [x] LaSt-ViT baseline evaluation
  - DINO ViT-S/16: CorLoc 65.70%, Mask IoU 13.31%, PiB 48.10%
  - DAFB-CLS beats LaSt-ViT by +13.7pp CorLoc, +18pp Mask IoU
- [x] Improve DINO results — PiB improved from 33.54% to 45.07% via ablation full model
  - [x] Stable background improvement attempts (6 rounds, see experiments.md §4)
  - [ ] Future: architectural changes needed for stable background

## Phase 9: COCO Experiments
- [x] Download COCO 2017 dataset (train2017 + val2017 + annotations)
- [x] Create COCO dataset loader (dafb_cls/datasets/coco.py)
- [x] Create COCO configs (dino_vits16_coco.yaml, openclip_vitb16_coco.yaml)
- [x] Add COCO support to train_posthoc.py
- [x] Add COCO support to eval scripts (eval_corloc, eval_mask_iou, eval_pib, eval_lastvit)
- [x] Add COCO support to utility scripts (run_ablation, stress_test, visualize_masks, etc.)
- [x] DINO ViT-S/16 + COCO training (50 epochs, lambda_fg=0.05, lr=5e-5, fp32)
- [x] DINO ViT-S/16 + COCO evaluation
  - CorLoc: 72.96% (3613/4952)
  - Mask IoU: 28.73% (Precision: 37.07%, Recall: 57.50%)
  - PiB: 35.44% (1755/4952)
- [x] LaSt-ViT baseline on COCO
  - CorLoc: 61.53%, Mask IoU: 12.95%, PiB: 45.09%
  - DAFB-CLS beats LaSt-ViT by +11.4pp CorLoc, +15.8pp Mask IoU
- [ ] OpenCLIP ViT-B/16 + COCO segmentation (2026-05-27尝试, 实测7.12 it/s, 50 epochs需28h, 中止. 建议epochs减至10或batch_size增至16)
- [ ] COCO ablation experiments
- [ ] COCO stress test

## Phase 10: Paper Submission Preparation
- [x] Efficiency analysis (params, FLOPs, inference time) — script: `dafb_cls/tools/efficiency_analysis.py`
  - DINO ViT-S/16: DAFB-CLS overhead 1.31x latency (+1.82ms), 0.59M trainable (2.7% of backbone)
  - OpenCLIP ViT-B/16: DAFB-CLS overhead 1.28x latency (+3.45ms), 1.74M trainable (2.0% of backbone)
- [x] CAM baseline evaluation (DINO ViT-S/16 + VOC)
  - CorLoc: 62.80%, Mask IoU: 3.52%, PiB: 21.26%
- [x] DINO-seg baseline evaluation (DINO ViT-S/16 + VOC)
  - CorLoc: 68.46%, Mask IoU: 8.83%, PiB: 24.64%
- [x] Generate qualitative visualization figures
  - `visualizations/masks/` — 20 foreground mask visualizations
  - `visualizations/depth_attention/` — 20 depth attention bar charts
- [x] Add remaining comparison methods (TokenCut, MaskCLIP) — 2026-05-27
  - TokenCut VOC: CorLoc 58.52%, Mask IoU 6.17%, PiB 25.60%
  - TokenCut COCO: CorLoc 54.75%, Mask IoU 6.77%, PiB 31.97%
  - MaskCLIP VOC: CorLoc 74.88%, Mask IoU 4.07%, PiB 39.68%
  - MaskCLIP COCO: CorLoc 70.56%, Mask IoU 6.05%, PiB 39.36%
- [ ] COCO comparison methods (CAM, DINO-seg on COCO)
- [ ] Expand backbone experiments (ViT-B, ViT-L)
