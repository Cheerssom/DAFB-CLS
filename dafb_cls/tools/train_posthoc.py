import os
import sys

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.dafb_cls_model import DAFBCLS
from dafb_cls.losses.mask_losses import CombinedMaskLoss, generate_pseudo_mask_from_patch_score
from dafb_cls.losses.decouple_loss import DecoupleLoss
from dafb_cls.losses.budget_loss import BudgetRegularizationLoss
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            if all(v.shape == values[0].shape for v in values):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        elif isinstance(values[0], list):
            result[key] = values
        else:
            result[key] = values
    return result


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DAFBCLS:
    model = DAFBCLS(cfg)
    model.freeze_backbone()
    return model


def build_optimizer(model: DAFBCLS, cfg: dict):
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def build_scheduler(optimizer, cfg: dict):
    scheduler_type = cfg.get("scheduler", "cosine")
    epochs = cfg.get("epochs", 50)
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        step_size = cfg.get("step_size", 20)
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def compute_main_loss(output: dict, batch: dict, cfg: dict, device) -> torch.Tensor:
    task = cfg.get("task", "classification")
    if task == "classification":
        labels = batch.get("label")
        if labels is not None:
            if isinstance(labels, list):
                labels = torch.tensor(labels, device=device)
            return F.cross_entropy(output["logits"], labels.to(device))
    elif task == "segmentation":
        seg_logits = output["seg_logits"]
        gt_mask = batch.get("mask")
        if gt_mask is not None:
            if isinstance(gt_mask, list):
                gt_mask = torch.stack([m.to(device) for m in gt_mask])
            gt_mask = gt_mask.to(device)
            B, N, C = seg_logits.shape
            grid_size = int(N ** 0.5)
            gt_mask_resized = F.interpolate(
                gt_mask.unsqueeze(1).float(), size=(grid_size, grid_size), mode="nearest"
            ).reshape(B, -1)
            return F.cross_entropy(seg_logits.float().permute(0, 2, 1), gt_mask_resized.long(), ignore_index=255)
    elif task == "object_discovery":
        score_map = output["score_map"]
        gt_mask = batch.get("mask")
        if gt_mask is not None:
            if isinstance(gt_mask, list):
                gt_mask = torch.stack([m.to(device) for m in gt_mask])
            gt_mask = gt_mask.to(device)
            B, N = score_map.shape
            grid_size = int(N ** 0.5)
            gt_binary = (gt_mask > 0).float()
            gt_mask_resized = F.interpolate(
                gt_binary.unsqueeze(1), size=(grid_size, grid_size), mode="nearest"
            ).reshape(B, -1)
            return F.binary_cross_entropy_with_logits(score_map, gt_mask_resized)
    return torch.tensor(0.0, device=output["C"].device)


def compute_fg_mask_loss(output: dict, batch: dict, cfg: dict, device) -> torch.Tensor:
    fg_logits = output.get("fg_logits")
    if fg_logits is None:
        return torch.tensor(0.0, device=device)

    gt_mask = batch.get("mask")
    if gt_mask is not None:
        if isinstance(gt_mask, list):
            gt_mask = torch.stack([m.to(device) for m in gt_mask])
        gt_mask = gt_mask.to(device)
        B, N = fg_logits.shape
        grid_size = int(N ** 0.5)
        gt_binary = (gt_mask > 0).float()
        gt_resized = F.interpolate(
            gt_binary.unsqueeze(1), size=(grid_size, grid_size), mode="nearest"
        ).reshape(B, -1)
        mask_loss_fn = CombinedMaskLoss()
        return mask_loss_fn(fg_logits, gt_resized)
    else:
        patch_score = output.get("foreground_score", output["foreground_mask"])
        pseudo_gt = generate_pseudo_mask_from_patch_score(patch_score, grid_size=int(patch_score.shape[1] ** 0.5))
        mask_loss_fn = CombinedMaskLoss()
        return mask_loss_fn(fg_logits, pseudo_gt)


def compute_ratio_align_loss(output: dict, batch: dict, cfg: dict, device) -> torch.Tensor:
    """MSE between predicted FG ratio and GT FG ratio, for background-dominated images."""
    fg_mask = output.get("foreground_mask")
    if fg_mask is None:
        return torch.tensor(0.0, device=device)

    gt_mask = batch.get("mask")
    if gt_mask is None:
        return torch.tensor(0.0, device=device)

    if isinstance(gt_mask, list):
        gt_mask = torch.stack([m.to(device) for m in gt_mask])
    else:
        gt_mask = gt_mask.to(device)

    B = fg_mask.shape[0]
    N = fg_mask.shape[1]
    grid_size = int(N ** 0.5)

    gt_binary = (gt_mask > 0).float()
    gt_resized = F.interpolate(
        gt_binary.unsqueeze(1), size=(grid_size, grid_size), mode="nearest"
    ).reshape(B, -1)

    pred_ratio = fg_mask.mean(dim=1)
    gt_ratio = gt_resized.mean(dim=1)

    # Scale by N so gradient isn't diluted by patch count
    return N * F.mse_loss(pred_ratio, gt_ratio)


def compute_tau_align_loss(output: dict, device) -> torch.Tensor:
    """MSE between learned tau and teacher tau, to train the tau predictor."""
    agg = output.get("agg_info", {})
    tau_learned = agg.get("tau_learned")
    tau_teacher = agg.get("tau_teacher")
    if tau_learned is None or tau_teacher is None:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(tau_learned, tau_teacher.detach())


def encode_text_features(model, cfg, device):
    backbone_type = cfg.get("backbone_type", "")
    if not backbone_type.startswith("openclip"):
        return None
    import open_clip
    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "coco":
        from dafb_cls.datasets.coco import COCO_CLASSES
        classes = list(COCO_CLASSES) + ["background"]
    else:
        from dafb_cls.datasets.voc import VOC_CLASSES
        classes = list(VOC_CLASSES) + ["background"]
    model_name = "ViT-B-16" if "vitb16" in backbone_type else "ViT-L-14"
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b88k")
    clip_model = clip_model.to(device)
    clip_model.eval()
    texts = [f"a photo of a {c}" for c in classes]
    tokenizer = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenizer)
        text_features = F.normalize(text_features, dim=-1)
    del clip_model
    torch.cuda.empty_cache()
    return text_features


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, cfg, device, epoch, text_features=None):
    model.train()
    model.freeze_backbone()

    ablation = cfg.get("ablation", {})
    disable_dafb = ablation.get("disable_dafb", False)

    lambda_fg = cfg.get("lambda_fg", 1.0) if not disable_dafb else 0.0
    lambda_decouple = cfg.get("lambda_decouple", 0.05) if not disable_dafb else 0.0
    lambda_budget = cfg.get("lambda_budget", 0.01) if not disable_dafb else 0.0
    lambda_ratio = cfg.get("lambda_ratio", 0.0)

    decouple_loss_fn = DecoupleLoss(method=cfg.get("decouple_method", "cosine_squared"))
    budget_loss_fn = BudgetRegularizationLoss(
        r_min=cfg.get("r_min", 0.1), r_max=cfg.get("r_max", 0.7),
    )

    total_loss = 0.0
    total_main = 0.0
    total_fg = 0.0
    total_decouple = 0.0
    total_budget = 0.0
    total_ratio = 0.0
    total_tau = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=cfg.get("amp", True)):
            output = model(images, text_features=text_features)

            loss_main = compute_main_loss(output, batch, cfg, device)
            loss_fg = compute_fg_mask_loss(output, batch, cfg, device)
            if isinstance(loss_fg, torch.Tensor):
                loss_fg = torch.where(torch.isfinite(loss_fg), loss_fg.clamp(max=50.0), torch.zeros_like(loss_fg))
            loss_decouple = decouple_loss_fn(output["C_F"], output["C_B"])
            loss_budget = budget_loss_fn(output["foreground_mask"])
            loss_ratio = compute_ratio_align_loss(output, batch, cfg, device)

            loss = loss_main + lambda_fg * loss_fg + lambda_decouple * loss_decouple + lambda_budget * loss_budget + lambda_ratio * loss_ratio

            # Skip NaN/Inf losses to prevent training collapse
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
                continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check for NaN/Inf gradients and skip if found
            has_nan_grad = False
            for p in model.get_trainable_params():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                optimizer.zero_grad()
                scaler.update()
                continue

            nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()

        # NaN-safe accumulation
        def _val(t):
            v = t.item() if isinstance(t, torch.Tensor) else t
            return v if (v == v and abs(v) < 1e30) else 0.0  # filter NaN and Inf

        total_loss += _val(loss)
        total_main += _val(loss_main)
        total_fg += _val(loss_fg)
        total_decouple += _val(loss_decouple)
        total_budget += _val(loss_budget)
        total_ratio += _val(loss_ratio)
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "main": f"{loss_main.item() if isinstance(loss_main, torch.Tensor) else loss_main:.4f}",
            "fg": f"{loss_fg.item():.4f}",
            "dec": f"{loss_decouple.item():.4f}",
        })

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(num_batches, 1),
        "main": total_main / max(num_batches, 1),
        "fg": total_fg / max(num_batches, 1),
        "decouple": total_decouple / max(num_batches, 1),
        "budget": total_budget / max(num_batches, 1),
        "ratio": total_ratio / max(num_batches, 1),
        "tau_align": total_tau / max(num_batches, 1),
    }


def evaluate(model, dataloader, cfg, device, text_features=None):
    model.eval()
    task = cfg.get("task", "classification")
    num_classes = cfg.get("num_classes", 21)
    grid_size = cfg.get("upsample_size") or (cfg.get("image_size", 224) // cfg.get("patch_size", 16))

    correct = 0
    total = 0
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)

            output = model(images, text_features=text_features)

            if task == "classification":
                labels = batch.get("label")
                if labels is not None:
                    if isinstance(labels, list):
                        labels = torch.tensor(labels, device=device)
                    preds = output["logits"].argmax(dim=1)
                    correct += (preds == labels.to(device)).sum().item()
                    total += labels.shape[0] if isinstance(labels, torch.Tensor) else len(labels)

            elif task == "segmentation":
                seg_logits = output.get("seg_logits")
                gt_mask = batch.get("mask")
                if seg_logits is not None and gt_mask is not None:
                    if isinstance(gt_mask, list):
                        gt_mask = torch.stack([m.to(device) for m in gt_mask])
                    else:
                        gt_mask = gt_mask.to(device)

                    B, N, C = seg_logits.shape
                    pred_labels = seg_logits.argmax(dim=-1)

                    gt_resized = F.interpolate(
                        gt_mask.unsqueeze(1).float(),
                        size=(grid_size, grid_size),
                        mode="nearest"
                    ).squeeze(1).reshape(B, -1).long()

                    for cls_idx in range(num_classes):
                        pred_cls = (pred_labels == cls_idx)
                        gt_cls = (gt_resized == cls_idx)
                        valid = (gt_resized != 255)
                        pred_cls = pred_cls & valid
                        gt_cls = gt_cls & valid
                        intersection[cls_idx] += (pred_cls & gt_cls).sum().float()
                        union[cls_idx] += (pred_cls | gt_cls).sum().float()

    metrics = {}
    if task == "classification" and total > 0:
        metrics["accuracy"] = correct / total

    if task == "segmentation":
        valid_classes = union > 0
        if valid_classes.any():
            iou = intersection[valid_classes] / union[valid_classes].clamp(min=1e-6)
            metrics["mIoU"] = iou.mean().item()
            metrics["per_class_iou"] = {
                str(i): iou[j].item() for j, i in enumerate(torch.where(valid_classes)[0].tolist())
            }

    if task == "object_discovery":
        pass  # metrics evaluated separately via eval_*.py scripts

    return metrics


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": {k: v for k, v in model.state_dict().items() if "extractor" not in k},
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    scaler = torch.amp.GradScaler("cuda") if cfg.get("amp", True) else None

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    dataset_type = cfg.get("dataset", "voc")
    if dataset_type == "voc":
        train_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("train_split", "train"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
        val_ds = VOCDataset(
            root=cfg["data_root"],
            year=cfg.get("voc_year", "2012"),
            split=cfg.get("val_split", "val"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    elif dataset_type == "coco":
        train_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_train_ann", os.path.join(cfg["data_root"], "annotations", "instances_train2017.json")),
            split=cfg.get("train_split", "train2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
        val_ds = COCODataset(
            root=cfg["data_root"],
            ann_file=cfg.get("coco_val_ann", os.path.join(cfg["data_root"], "annotations", "instances_val2017.json")),
            split=cfg.get("val_split", "val2017"),
            image_size=cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    epochs = cfg.get("epochs", 50)
    save_dir = cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    text_features = encode_text_features(model, cfg, device)
    if text_features is not None:
        print(f"Encoded text features: {text_features.shape}")

    for epoch in range(start_epoch, epochs):
        model.train()  # ensure train mode after eval
        metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch, text_features=text_features)
        print(f"Epoch {epoch}: {metrics}")

        if (epoch + 1) % cfg.get("eval_interval", 5) == 0:
            eval_metrics = evaluate(model, val_loader, cfg, device, text_features=text_features)
            print(f"Eval at epoch {epoch}: {eval_metrics}")

        if (epoch + 1) % cfg.get("save_interval", 10) == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"epoch_{epoch}.pt"))

    save_checkpoint(model, optimizer, epochs - 1, os.path.join(save_dir, "final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
