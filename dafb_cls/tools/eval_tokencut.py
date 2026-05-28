"""
TokenCut baseline for unsupervised object discovery.

TokenCut (Wang et al., 2023):
  1. Extract CLS-to-patch self-attention from the last ViT layer
  2. Build a bipartite graph between CLS and patch tokens
  3. Apply normalized cuts (NCut) to bipartition into foreground/background
  4. Select the cluster containing the most-attended patches

Metrics: CorLoc, Mask IoU, PiB.

Usage:
    python -m dafb_cls.tools.eval_tokencut --config dafb_cls/configs/dino_vits16_voc.yaml
    python -m dafb_cls.tools.eval_tokencut --config dafb_cls/configs/dino_vits16_coco.yaml
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset
from dafb_cls.tools.train_posthoc import collate_fn


class TokenCutBaseline:
    """TokenCut: normalized-cut-based unsupervised object discovery."""

    def __init__(self, backbone_type="dino_vits16"):
        import timm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = {
            "dino_vits16": "vit_small_patch16_224",
            "dino_vitb16": "vit_base_patch16_224",
            "deit_vits16": "deit_small_patch16_224",
            "deit_vitb16": "deit_base_patch16_224",
        }.get(backbone_type, "vit_small_patch16_224")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_attention_map(self, images):
        """Extract CLS-to-patch attention from the last layer."""
        with torch.no_grad():
            x = self.model.patch_embed(images)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.model.pos_embed

            for blk in self.model.blocks[:-1]:
                x = blk(x)

            # Last block: manual attention computation
            last_blk = self.model.blocks[-1]
            norm1 = last_blk.norm1(x)
            qkv = last_blk.attn.qkv(norm1).reshape(
                x.shape[0], x.shape[1], 3, last_blk.attn.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            scale = q.shape[-1] ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N+1, N+1]
            attn = attn.softmax(dim=-1)

            # CLS-to-patch attention, averaged over heads
            cls_attn = attn[:, :, 0, 1:]  # [B, heads, N]
            cls_attn = cls_attn.mean(dim=1)  # [B, N]

            return cls_attn

    def get_patch_features(self, images):
        """Extract patch features from the last layer."""
        with torch.no_grad():
            x = self.model.patch_embed(images)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.model.pos_embed

            for blk in self.model.blocks:
                x = blk(x)

            # Remove CLS token, keep patch tokens
            patch_feats = x[:, 1:, :]  # [B, N, D]
            return patch_feats

    @staticmethod
    def ncut_bipartition(attn_weights, patch_features, tau=0.3):
        """
        TokenCut normalized-cut bipartition.

        Builds a bipartite graph: CLS node connected to patches with weights
        from attention. Uses NCut to find the optimal bipartition.

        Args:
            attn_weights: [N] CLS-to-patch attention weights
            patch_features: [N, D] patch feature vectors
            tau: threshold ratio for binarizing attention (relative to max)

        Returns:
            fg_mask: [N] binary foreground mask
        """
        N = attn_weights.shape[0]
        device = attn_weights.device

        # Build affinity matrix from attention weights
        # A[i,j] = attn[i] * attn[j] * cosine_sim(patch_i, patch_j)
        # Simplified: use attention as node weights, cosine similarity as edge weights

        # Normalize attention weights
        a = attn_weights / (attn_weights.sum() + 1e-8)  # [N]

        # Compute pairwise cosine similarity between patches
        patch_norm = F.normalize(patch_features.float(), dim=-1)  # [N, D]
        sim_matrix = patch_norm @ patch_norm.T  # [N, N]

        # Affinity: W_ij = a_i * a_j * max(sim_ij, 0)
        sim_matrix = sim_matrix.clamp(min=0)
        W = a.unsqueeze(1) * a.unsqueeze(0) * sim_matrix  # [N, N]

        # Degree matrix
        D = W.sum(dim=1)  # [N]
        D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)  # [N]

        # Normalized cut: solve generalized eigenvalue problem
        # L_sym = I - D^{-1/2} W D^{-1/2}
        # Second smallest eigenvector gives the bipartition
        D_inv_sqrt_diag = torch.diag(D_inv_sqrt)
        L_sym = torch.eye(N, device=device) - D_inv_sqrt_diag @ W @ D_inv_sqrt_diag

        # Use power iteration for the second eigenvector (Fiedler vector)
        # For efficiency, use torch.linalg.eigh on CPU for small N
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym.cpu())
            # Second smallest eigenvalue's eigenvector
            fiedler = eigenvectors[:, 1].to(device)  # [N]
        except Exception:
            # Fallback: use attention thresholding
            threshold = attn_weights.mean() + tau * attn_weights.std()
            return (attn_weights > threshold).float()

        # Bipartition based on sign of Fiedler vector
        # The cluster with higher mean attention = foreground
        fg_mask = (fiedler > 0).float()
        if fg_mask.sum() == 0 or fg_mask.sum() == N:
            # Degenerate case: fallback to threshold
            threshold = attn_weights.mean() + tau * attn_weights.std()
            return (attn_weights > threshold).float()

        # Ensure the higher-attention cluster is foreground
        fg_attn_mean = (attn_weights * fg_mask).sum() / (fg_mask.sum() + 1e-8)
        bg_attn_mean = (attn_weights * (1 - fg_mask)).sum() / ((1 - fg_mask).sum() + 1e-8)
        if bg_attn_mean > fg_attn_mean:
            fg_mask = 1 - fg_mask

        return fg_mask

    def get_foreground_mask(self, images, tau=0.3):
        """Get TokenCut foreground mask."""
        attn = self.get_attention_map(images)  # [B, N]
        patch_feats = self.get_patch_features(images)  # [B, N, D]

        B, N = attn.shape
        masks = []
        for i in range(B):
            mask = self.ncut_bipartition(attn[i], patch_feats[i], tau=tau)
            masks.append(mask)

        return torch.stack(masks), attn  # [B, N], [B, N]


def evaluate_corloc_tokencut(model, dataloader, cfg, device, tau=0.3):
    """CorLoc evaluation using TokenCut."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="CorLoc"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        bboxes = batch.get("bboxes")
        if bboxes is None:
            continue

        mask, attn = model.get_foreground_mask(images, tau=tau)  # [B, N]
        B = mask.shape[0]

        for i in range(B):
            bbox_list = bboxes[i] if isinstance(bboxes, list) else bboxes[i]
            if hasattr(bbox_list, 'shape') and len(bbox_list.shape) > 0 and bbox_list.shape[0] == 0:
                continue
            if isinstance(bbox_list, torch.Tensor):
                bbox_list = bbox_list.tolist()
            if len(bbox_list) == 0:
                continue

            fg_idx = mask[i].nonzero(as_tuple=True)[0]
            if len(fg_idx) == 0:
                fg_idx = attn[i].topk(max(1, attn[i].shape[0] // 4)).indices

            fg_rows = fg_idx // grid_size
            fg_cols = fg_idx % grid_size
            fg_cx = (fg_cols.float() + 0.5).mean() * patch_size
            fg_cy = (fg_rows.float() + 0.5).mean() * patch_size

            hit = False
            for bbox in bbox_list:
                if isinstance(bbox, (list, tuple)):
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if x1 <= fg_cx <= x2 and y1 <= fg_cy <= y2:
                    hit = True
                    break
            if hit:
                correct += 1
            total += 1

    corloc = correct / max(total, 1)
    print(f"TokenCut CorLoc: {corloc:.4f} ({correct}/{total})")
    return corloc


def evaluate_mask_iou_tokencut(model, dataloader, cfg, device, tau=0.3):
    """Mask IoU evaluation using TokenCut."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)

    all_iou, all_precision, all_recall = [], [], []

    for batch in tqdm(dataloader, desc="Mask IoU"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        gt_masks = batch.get("mask")
        if gt_masks is None:
            continue
        if isinstance(gt_masks, list):
            gt_masks = torch.stack([m.to(device) for m in gt_masks])
        else:
            gt_masks = gt_masks.to(device)

        mask, _ = model.get_foreground_mask(images, tau=tau)  # [B, N]
        B = mask.shape[0]

        for i in range(B):
            pred_2d = mask[i].reshape(grid_size, grid_size).float()
            pred_full = F.interpolate(
                pred_2d.unsqueeze(0).unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            gt_i = gt_masks[i]
            if gt_i.max() > 1:
                gt_i = (gt_i > 0).float()
            gt_binary = gt_i.float()

            pred_binary = (pred_full > 0.5).float()
            intersection = (pred_binary * gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum() - intersection

            iou = intersection / (union + 1e-6)
            precision = intersection / (pred_binary.sum() + 1e-6)
            recall = intersection / (gt_binary.sum() + 1e-6)
            all_iou.append(iou.item())
            all_precision.append(precision.item())
            all_recall.append(recall.item())

    mean_iou = np.mean(all_iou) if all_iou else 0.0
    mean_p = np.mean(all_precision) if all_precision else 0.0
    mean_r = np.mean(all_recall) if all_recall else 0.0
    print(f"TokenCut Mask IoU: {mean_iou:.4f}  (Precision: {mean_p:.4f}, Recall: {mean_r:.4f})")
    return {"mean_iou": mean_iou, "mean_precision": mean_p, "mean_recall": mean_r}


def evaluate_pib_tokencut(model, dataloader, cfg, device, tau=0.3):
    """PiB evaluation using TokenCut CLS-patch similarity."""
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    patch_size = cfg.get("patch_size", 16)

    hits, total = 0, 0

    for batch in tqdm(dataloader, desc="PiB"):
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        bboxes = batch.get("bboxes")
        if bboxes is None:
            continue

        attn = model.get_attention_map(images)  # [B, N]
        patch_feats = model.get_patch_features(images)  # [B, N, D]

        # CLS token ≈ attention-weighted average of patch features
        attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        cls_token = (attn_norm.unsqueeze(-1) * patch_feats).sum(dim=1)  # [B, D]

        # Most similar patch to CLS token
        cls_norm = F.normalize(cls_token, dim=-1).unsqueeze(1)
        patch_norm = F.normalize(patch_feats, dim=-1)
        similarity = (patch_norm * cls_norm).sum(dim=-1)  # [B, N]

        B = similarity.shape[0]
        for i in range(B):
            bbox_list = bboxes[i] if isinstance(bboxes, list) else bboxes[i]
            if hasattr(bbox_list, 'shape') and len(bbox_list.shape) > 0 and bbox_list.shape[0] == 0:
                continue
            if isinstance(bbox_list, torch.Tensor):
                bbox_list = bbox_list.tolist()
            if len(bbox_list) == 0:
                continue

            max_idx = similarity[i].argmax().item()
            row = max_idx // grid_size
            col = max_idx % grid_size
            cx = (col + 0.5) * patch_size
            cy = (row + 0.5) * patch_size

            hit = False
            for bbox in bbox_list:
                if isinstance(bbox, (list, tuple)):
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1

    pib = hits / max(total, 1)
    print(f"TokenCut PiB: {pib:.4f} ({hits}/{total})")
    return pib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML (e.g. dino_vits16_voc.yaml)")
    parser.add_argument("--tau", type=float, default=0.3,
                        help="NCut bipartition threshold (default=0.3)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    backbone_type = cfg.get("backbone_type", "dino_vits16")
    print(f"Device: {device}")
    print(f"Backbone: {backbone_type}")
    print(f"NCut tau: {args.tau}")

    model = TokenCutBaseline(backbone_type=backbone_type)

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 16),
        shuffle=False, num_workers=cfg.get("num_workers", 0),
        pin_memory=True, collate_fn=collate_fn,
    )

    print(f"\n{'='*60}")
    print(f"TokenCut Baseline Evaluation ({dataset_type.upper()})")
    print(f"{'='*60}")

    corloc = evaluate_corloc_tokencut(model, val_loader, cfg, device, tau=args.tau)
    mask_iou = evaluate_mask_iou_tokencut(model, val_loader, cfg, device, tau=args.tau)
    pib = evaluate_pib_tokencut(model, val_loader, cfg, device, tau=args.tau)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CorLoc:   {corloc*100:.2f}%")
    print(f"  Mask IoU: {mask_iou['mean_iou']*100:.2f}%")
    print(f"  PiB:      {pib*100:.2f}%")


if __name__ == "__main__":
    main()
