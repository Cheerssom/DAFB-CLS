import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset
from dafb_cls.tools.train_posthoc import (
    build_model, build_optimizer, build_scheduler,
    train_one_epoch, save_checkpoint, collate_fn, load_config,
)
from dafb_cls.tools.eval_corloc import evaluate_corloc
from dafb_cls.tools.eval_mask_iou import evaluate_mask_iou
from dafb_cls.tools.eval_pib import evaluate_pib

SUMMARY_DIR = "checkpoints/ablation_dino"
SUMMARY_PATH = os.path.join(SUMMARY_DIR, "results.yaml")


def merge_ablation_config(base_cfg: dict, variant: dict) -> dict:
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["save_dir"] = os.path.join(SUMMARY_DIR, variant["name"])
    ablation = variant.copy()
    del ablation["name"]
    ablation.pop("description", None)
    existing_ablation = cfg.get("ablation", {})
    existing_ablation.update(ablation)
    cfg["ablation"] = existing_ablation
    return cfg


def load_previous_results():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r") as f:
            try:
                data = yaml.safe_load(f) or []
            except yaml.constructor.ConstructorError:
                # Corrupted file (numpy tags), skip
                return []
        return data
    return []


def save_results(results):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    # Convert any numpy types to native Python to avoid YAML serialization errors
    clean = []
    for r in results:
        clean.append({k: float(v) if hasattr(v, 'item') else v for k, v in r.items()})
    with open(SUMMARY_PATH, "w") as f:
        yaml.dump(clean, f)


def print_table(results):
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Variant':<20} {'CorLoc%':>10} {'MaskIoU%':>10} {'PiB%':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['variant']:<20} {r['corloc']:>10.2f} {r['mask_iou']:>10.2f} {r['pib']:>10.2f}")
    print("-" * 50)


def maybe_label_to_int(label, variant_name: str):
    """Fix labels to integers for cross_entropy compatibility."""
    if label is None:
        return None
    if isinstance(label, torch.Tensor):
        if label.dtype in (torch.int32, torch.int64, torch.long):
            return label
        return label.long()
    if isinstance(label, list):
        try:
            return torch.tensor([int(l) for l in label], dtype=torch.long)
        except (ValueError, TypeError):
            return torch.tensor(label, dtype=torch.long)
    return label


def train_variant(variant: dict, base_cfg: dict, train_loader, val_loader, device):
    variant_name = variant["name"]
    cfg = merge_ablation_config(base_cfg, variant)

    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"{'='*60}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Find latest checkpoint if resuming
    start_epoch = 0
    latest_ckpt = None
    ckpt_files = sorted(
        [f for f in os.listdir(cfg["save_dir"]) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda x: int(x.replace("epoch_", "").replace(".pt", ""))
    )
    if ckpt_files:
        latest_ckpt = os.path.join(cfg["save_dir"], ckpt_files[-1])
        start_epoch = int(ckpt_files[-1].replace("epoch_", "").replace(".pt", "")) + 1

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if latest_ckpt:
        print(f"  Resuming from {latest_ckpt} (epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and hasattr(scheduler, 'last_epoch'):
            scheduler.last_epoch = start_epoch - 1
    else:
        print(f"  Starting from scratch")

    scaler = torch.amp.GradScaler("cuda") if cfg.get("amp", True) else None

    epochs = cfg.get("epochs", 50)
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch,
            text_features=None,
        )
        print(f"  Epoch {epoch}: loss={train_metrics['loss']:.4f} "
              f"main={train_metrics['main']:.4f} dec={train_metrics['decouple']:.4f}")

        if (epoch + 1) % cfg.get("save_interval", 10) == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(cfg["save_dir"], f"epoch_{epoch}.pt"))

    final_path = os.path.join(cfg["save_dir"], "final.pt")
    save_checkpoint(model, optimizer, epochs - 1, final_path)

    print(f"\n  Evaluating {variant_name}...")
    corloc = evaluate_corloc(model, val_loader, cfg, device)
    mask_iou_metrics = evaluate_mask_iou(model, val_loader, cfg, device, text_features=None)
    pib = evaluate_pib(model, val_loader, cfg, device)

    return {
        "variant": variant_name,
        "corloc": round(corloc * 100, 2),
        "mask_iou": round(mask_iou_metrics.get("mean_iou", 0) * 100, 2),
        "pib": round(pib * 100, 2),
        "ckpt": final_path,
    }


def run_ablations(config_path: str, target_variant: str = None):
    base_cfg = load_config(config_path)
    all_variants = base_cfg.get("ablation", {}).get("variants", [])

    if not all_variants:
        print("No ablation variants found in config.")
        return

    if target_variant:
        variants = [v for v in all_variants if v["name"] == target_variant]
        if not variants:
            print(f"Variant '{target_variant}' not found. Available:")
            for v in all_variants:
                print(f"  - {v['name']}")
            return
    else:
        variants = all_variants

    device = torch.device(base_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Running {len(variants)} variant(s):")
    for v in variants:
        print(f"  - {v['name']}")

    # Build dataloaders (shared across variants)
    dataset_type = base_cfg.get("dataset", "voc")
    if dataset_type == "coco":
        train_ds = COCODataset(
            root=base_cfg["data_root"],
            ann_file=base_cfg.get("coco_train_ann", os.path.join(base_cfg["data_root"], "annotations", "instances_train2017.json")),
            split=base_cfg.get("train_split", "train2017"),
            image_size=base_cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
        val_ds = COCODataset(
            root=base_cfg["data_root"],
            ann_file=base_cfg.get("coco_val_ann", os.path.join(base_cfg["data_root"], "annotations", "instances_val2017.json")),
            split=base_cfg.get("val_split", "val2017"),
            image_size=base_cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    else:
        train_ds = VOCDataset(
            root=base_cfg["data_root"],
            year=base_cfg.get("voc_year", "2012"),
            split=base_cfg.get("train_split", "train"),
            image_size=base_cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
        val_ds = VOCDataset(
            root=base_cfg["data_root"],
            year=base_cfg.get("voc_year", "2012"),
            split=base_cfg.get("val_split", "val"),
            image_size=base_cfg.get("image_size", 224),
            return_mask=True,
            return_bbox=True,
        )
    train_loader = DataLoader(
        train_ds, batch_size=base_cfg.get("batch_size", 16),
        shuffle=True, num_workers=base_cfg.get("num_workers", 0),
        pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=base_cfg.get("batch_size", 16),
        shuffle=False, num_workers=base_cfg.get("num_workers", 0),
        pin_memory=True, collate_fn=collate_fn,
    )

    # Load existing results, update as we go
    prev_results = load_previous_results()
    prev_map = {r["variant"]: r for r in prev_results}
    all_results = prev_results.copy()

    for variant in variants:
        result = train_variant(variant, base_cfg, train_loader, val_loader, device)
        print(f"  -> CorLoc={result['corloc']}%, MaskIoU={result['mask_iou']}%, PiB={result['pib']}%")

        # Update in-place
        if result["variant"] in prev_map:
            for i, r in enumerate(all_results):
                if r["variant"] == result["variant"]:
                    all_results[i] = result
                    break
        else:
            all_results.append(result)
        save_results(all_results)

    print_table(all_results)
    print(f"\nResults saved to {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, default=None,
                        help="Run a single variant (e.g. full, no_budget, baseline_dino)")
    parser.add_argument("--list", action="store_true",
                        help="List available variants and exit")
    args = parser.parse_args()

    if args.list:
        cfg = load_config(args.config)
        variants = cfg.get("ablation", {}).get("variants", [])
        for v in variants:
            print(f"  {v['name']}: {v.get('description', '')}")
        return

    run_ablations(args.config, target_variant=args.variant)


if __name__ == "__main__":
    main()
