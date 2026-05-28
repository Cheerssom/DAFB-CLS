"""
Download and extract COCO 2017 train/val images + annotations.

Usage:
    python scripts/download_coco2017.py              # download all
    python scripts/download_coco2017.py --val-only   # val only (smaller, for quick experiments)
"""

import os
import argparse
import zipfile
import urllib.request
import shutil

COCO_BASE = "http://images.cocodataset.org/zips"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "coco2017")

FILES = {
    "train2017.zip": f"{COCO_BASE}/train2017.zip",
    "val2017.zip": f"{COCO_BASE}/val2017.zip",
    "annotations_trainval2017.zip": ANN_URL,
}


def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest}")


def extract_zip(zip_path: str, dest_dir: str):
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017")
    parser.add_argument("--val-only", action="store_true", help="Only download val2017 + annotations")
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip files after extraction")
    args = parser.parse_args()

    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_dir = os.path.join(TARGET_DIR, "zips")
    os.makedirs(zip_dir, exist_ok=True)

    files_to_download = {
        "val2017.zip": FILES["val2017.zip"],
        "annotations_trainval2017.zip": FILES["annotations_trainval2017.zip"],
    }
    if not args.val_only:
        files_to_download["train2017.zip"] = FILES["train2017.zip"]

    # Download
    for fname, url in files_to_download.items():
        zip_path = os.path.join(zip_dir, fname)
        download_file(url, zip_path)

    # Extract
    for fname in files_to_download:
        zip_path = os.path.join(zip_dir, fname)
        extract_zip(zip_path, TARGET_DIR)

    # Cleanup
    if not args.keep_zips:
        print("  Removing zip files ...")
        shutil.rmtree(zip_dir, ignore_errors=True)

    # Verify
    print("\nVerification:")
    for subdir in ["annotations", "val2017"]:
        path = os.path.join(TARGET_DIR, subdir)
        if os.path.exists(path):
            n = len(os.listdir(path))
            print(f"  {subdir}: {n} files OK")
        else:
            print(f"  {subdir}: MISSING!")

    if not args.val_only:
        train_path = os.path.join(TARGET_DIR, "train2017")
        if os.path.exists(train_path):
            n = len(os.listdir(train_path))
            print(f"  train2017: {n} files OK")
        else:
            print(f"  train2017: MISSING!")

    ann_path = os.path.join(TARGET_DIR, "annotations", "instances_val2017.json")
    if os.path.exists(ann_path):
        print(f"\nAnnotation file: {ann_path}")
    else:
        print(f"\nWARNING: Annotation file not found at {ann_path}")

    print("\nDone!")
    print(f"  Dataset root: {os.path.abspath(TARGET_DIR)}")


if __name__ == "__main__":
    main()
