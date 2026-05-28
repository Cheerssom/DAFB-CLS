import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.datasets.voc import VOCDataset, VOC_CLASSES
from dafb_cls.datasets.coco import COCODataset, COCO_CLASSES


VOC_CLASSES_WITH_BG = ["background"] + VOC_CLASSES
COCO_CLASSES_WITH_BG = ["background"] + COCO_CLASSES


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_text_features(model_backbone, class_names, device):
    import open_clip
    tokenizer = open_clip.tokenize(class_names).to(device)
    with torch.no_grad():
        text_features = model_backbone.encode_text(tokenizer)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def compute_ovseg_miou(pred_logits: torch.Tensor, gt_mask: torch.Tensor, num_classes: int) -> dict:
    pred_labels = pred_logits.argmax(dim=-1)
    ious = []
    for cls_idx in range(num_classes):
        pred_cls = (pred_labels == cls_idx).float()
        gt_cls = (gt_mask == cls_idx).float()
        intersection = (pred_cls * gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum() - intersection
        if union > 0:
            ious.append((intersection / union).item())
    return {
        "miou": np.mean(ious) if ious else 0.0,
        "class_ious": {VOC_CLASSES_WITH_BG[i]: ious[i] if i < len(ious) else 0.0 for i in range(num_classes)},
    }


def evaluate_ovseg(model, dataloader, cfg, device):
    model.eval()
    grid_size = cfg.get("image_size", 224) // cfg.get("patch_size", 16)
    image_size = cfg.get("image_size", 224)

    text_features = None
    if hasattr(model, "extractor"):
        backbone_type = cfg.get("backbone_type", "")
        if backbone_type.startswith("openclip"):
            import open_clip
            model_name = "ViT-B-16" if "vitb16" in backbone_type else "ViT-L-14"
            clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b88k")
            clip_model = clip_model.to(device)
            clip_model.eval()
            text_features = get_text_features(clip_model, VOC_CLASSES_WITH_BG, device)
            del clip_model
            torch.cuda.empty_cache()

    all_ious = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating OVSeg"):
            images = batch["image"].to(device)
            gt_masks = batch.get("mask")
            if gt_masks is None:
                continue
            gt_masks = gt_masks.to(device)

            output = model(images, text_features=text_features)
            seg_logits = output.get("seg_logits")

            if seg_logits is None:
                continue

            B = seg_logits.shape[0]
            num_classes = seg_logits.shape[-1]
            for i in range(B):
                logit_2d = seg_logits[i].reshape(grid_size, grid_size, num_classes)
                logit_full = F.interpolate(
                    logit_2d.permute(2, 0, 1).unsqueeze(0),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                gt_i = gt_masks[i]
                if gt_i.dim() == 2 and gt_i.max() > 1:
                    gt_i_resized = F.interpolate(
                        gt_i.unsqueeze(0).unsqueeze(0).float(),
                        size=(image_size, image_size),
                        mode="nearest",
                    ).squeeze().long()
                else:
                    gt_i_resized = gt_i

                metrics = compute_ovseg_miou(logit_full.permute(1, 2, 0), gt_i_resized, num_classes)
                all_ious.append(metrics["miou"])

    mean_miou = np.mean(all_ious) if all_ious else 0.0
    print(f"OVSeg mIoU: {mean_miou:.4f}")
    return {"miou": mean_miou}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = DAFBCLS(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.freeze_backbone()
    model = model.to(device)

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    else:
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    evaluate_ovseg(model, val_loader, cfg, device)


if __name__ == "__main__":
    main()
