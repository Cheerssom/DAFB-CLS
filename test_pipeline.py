import sys
import os

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

print("=" * 60)
print("STEP 1: Test VOC Dataset Loading")
print("=" * 60)

from dafb_cls.datasets.voc import VOCDataset

ds_train = VOCDataset(
    root="E:/datasets",
    year="2012",
    split="train",
    image_size=224,
    return_mask=True,
    return_bbox=True,
)
print(f"  Train set size: {len(ds_train)}")

sample = ds_train[0]
print(f"  Image shape: {sample['image'].shape}")
print(f"  Image ID: {sample['image_id']}")
print(f"  Mask shape: {sample['mask'].shape}")
print(f"  Bboxes shape: {sample['bboxes'].shape}")
print(f"  Labels: {sample['labels']}")
print("  [OK] VOC Dataset loads successfully!\n")

print("=" * 60)
print("STEP 2: Smoke Test - Dummy Forward (DINO ViT-S/16)")
print("=" * 60)

from dafb_cls.models.feature_extractor import MultiLayerFeatureExtractor
from dafb_cls.models.dual_cls import DualCLSAggregator
from dafb_cls.models.depth_attention import ForegroundBackgroundDepthAttention
from dafb_cls.models.fusion import TaskAdaptiveFusionHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

B = 2
L = 4
N = 196
D = 384

print(f"  Creating dummy input: B={B}, L={L}, N={N}, D={D}")

patch_features = torch.randn(B, L, N, D, device=device)
cls_features = torch.randn(B, D, device=device)

print("  Testing DualCLSAggregator...")
dual_cls = DualCLSAggregator(
    dim=D, grid_size=14, hidden_dim=128,
    frequency_sigma=0.25,
    depth_method="cosine",
    semantic_method="patch_cls_similarity",
    spatial_kernel=3,
    tau_from_global=True,
).to(device)

B_F, B_B, agg_info = dual_cls(patch_features, cls_features=cls_features)
print(f"    B_F shape: {B_F.shape}")
print(f"    B_B shape: {B_B.shape}")
print(f"    fg_mask shape: {agg_info['foreground_mask'].shape}")
print(f"    fg_ratio: {agg_info['fg_ratio'].mean():.3f}")
print("    [OK] DualCLSAggregator works!\n")

print("  Testing ForegroundBackgroundDepthAttention...")
depth_attn = ForegroundBackgroundDepthAttention(dim=D, zero_init=True).to(device)
C_F, C_B, beta_F, beta_B = depth_attn(B_F, B_B)
print(f"    C_F shape: {C_F.shape}")
print(f"    C_B shape: {C_B.shape}")
print(f"    beta_F shape: {beta_F.shape}")
print(f"    beta_B shape: {beta_B.shape}")
print(f"    beta_F (should sum~1): {beta_F[0].tolist()}")
print(f"    beta_B (should sum~1): {beta_B[0].tolist()}")
print("    [OK] DepthAttention works!\n")

print("  Testing TaskAdaptiveFusionHead...")
fusion = TaskAdaptiveFusionHead(dim=D, hidden_dim=128).to(device)
global_feat = cls_features
C, gate = fusion(C_F, C_B, global_feat)
print(f"    C shape: {C.shape}")
print(f"    gate shape: {gate.shape}")
print(f"    gate value: {gate[0].item():.3f}")
print("    [OK] Fusion works!\n")

print("=" * 60)
print("STEP 3: Smoke Test - Loss Computation")
print("=" * 60)

from dafb_cls.losses.decouple_loss import DecoupleLoss
from dafb_cls.losses.budget_loss import BudgetRegularizationLoss
from dafb_cls.losses.mask_losses import CombinedMaskLoss

decouple_fn = DecoupleLoss(method="cosine_squared")
loss_decouple = decouple_fn(C_F, C_B)
print(f"  Decouple loss: {loss_decouple.item():.4f}")

budget_fn = BudgetRegularizationLoss(r_min=0.1, r_max=0.7)
loss_budget = budget_fn(agg_info["foreground_mask"])
print(f"  Budget loss: {loss_budget.item():.4f}")

fg_mask = agg_info["foreground_mask"]
gt_mask = torch.rand(B, N, device=device).round()
mask_loss_fn = CombinedMaskLoss()
loss_mask = mask_loss_fn(fg_mask, gt_mask)
print(f"  Mask loss: {loss_mask.item():.4f}")

total_loss = loss_decouple + loss_budget + loss_mask
total_loss.backward()
print(f"  Total loss: {total_loss.item():.4f}")
print("  [OK] All losses computed and backward pass works!\n")

print("=" * 60)
print("STEP 4: Smoke Test - Full DAFBCLS Model (with real backbone)")
print("=" * 60)

from dafb_cls.models.dafb_cls_model import DAFBCLS

cfg = {
    "backbone_type": "dino_vits16",
    "layer_indices": [3, 6, 9, 12],
    "image_size": 224,
    "patch_size": 16,
    "hidden_dim": 128,
    "fusion_hidden_dim": 128,
    "head_hidden_dim": 256,
    "task": "object_discovery",
    "num_classes": 20,
    "frequency_sigma": 0.25,
    "depth_method": "cosine",
    "semantic_method": "patch_cls_similarity",
    "spatial_kernel": 3,
    "tau_from_global": True,
    "depth_attn_zero_init": True,
}

print("  Building DAFBCLS with DINO ViT-S/16...")
model = DAFBCLS(cfg).to(device)
model.freeze_backbone()

trainable_params = sum(p.numel() for p in model.get_trainable_params())
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params / 1e6:.1f}M")
print(f"  Trainable params: {trainable_params / 1e6:.2f}M")

dummy_img = torch.randn(1, 3, 224, 224, device=device)
print("  Running forward pass...")
with torch.no_grad():
    output = model(dummy_img)

print(f"  C shape: {output['C'].shape}")
print(f"  C_F shape: {output['C_F'].shape}")
print(f"  C_B shape: {output['C_B'].shape}")
print(f"  gate: {output['gate'][0].item():.3f}")
print(f"  fg_mask shape: {output['foreground_mask'].shape}")
print(f"  fg_ratio: {output['foreground_mask'][0].mean():.3f}")
print(f"  score_map shape: {output['score_map'].shape}")
print(f"  beta_F: {output['beta_F'][0].tolist()}")
print(f"  beta_B: {output['beta_B'][0].tolist()}")
print("  [OK] Full model forward pass works!\n")

print("=" * 60)
print("STEP 5: Test Training Loop (1 batch)")
print("=" * 60)

optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4)
optimizer.zero_grad()

dummy_img = torch.randn(2, 3, 224, 224, device=device)
output = model(dummy_img)

loss_main = output["score_map"].mean()
loss = loss_main + 0.05 * decouple_fn(output["C_F"], output["C_B"]) + 0.01 * budget_fn(output["foreground_mask"])
loss.backward()
optimizer.step()
print(f"  Training loss: {loss.item():.4f}")
print("  [OK] Training step works!\n")

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
