import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dafb_cls.models.feature_extractor import MultiLayerFeatureExtractor
from dafb_cls.datasets.voc import VOCDataset
from dafb_cls.datasets.coco import COCODataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="cached_features")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    extractor = MultiLayerFeatureExtractor(
        backbone_type=cfg.get("backbone_type", "openclip_vitb16"),
        layer_indices=cfg.get("layer_indices", [3, 6, 9, 12]),
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 16),
    )
    extractor.build_backbone().to(device)
    extractor.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_type = cfg.get("dataset", "voc")
    for split in ["train", "val"]:
        if dataset_type == "coco":
            ann_file = cfg.get("coco_train_ann" if split == "train" else "coco_val_ann",
                               os.path.join(cfg["data_root"], "annotations", f"instances_{split}2017.json"))
            ds = COCODataset(
                root=cfg["data_root"],
                ann_file=ann_file,
                split=f"{split}2017",
                image_size=cfg.get("image_size", 224),
                return_mask=True,
                return_bbox=True,
            )
        else:
            ds = VOCDataset(
                root=cfg["data_root"],
                year=cfg.get("voc_year", "2012"),
                split=split,
                image_size=cfg.get("image_size", 224),
                return_mask=True,
                return_bbox=True,
            )
        loader = DataLoader(
            ds,
            batch_size=cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Caching {split}"):
                images = batch["image"].to(device)
                patch_features, cls_features = extractor(images, return_cls=True)

                for i in range(images.shape[0]):
                    img_id = batch["image_id"][i] if isinstance(batch["image_id"], list) else batch["image_id"]
                    if isinstance(img_id, (list, tuple)):
                        img_id = img_id[i]

                    save_dict = {
                        "patch_features": patch_features[i].cpu(),
                        "cls_features": cls_features[i].cpu() if cls_features is not None else None,
                    }
                    if "mask" in batch:
                        save_dict["mask"] = batch["mask"][i].cpu()
                    if "bboxes" in batch:
                        save_dict["bboxes"] = batch["bboxes"][i].cpu()

                    torch.save(save_dict, os.path.join(split_dir, f"{img_id}.pt"))

        print(f"Saved {len(ds)} feature files to {split_dir}")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
