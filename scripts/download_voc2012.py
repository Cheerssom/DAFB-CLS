import os
import tarfile
import urllib.request
import shutil

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
TAR_PATH = os.path.join(TARGET_DIR, "VOCtrainval_11-May-2012.tar")
EXTRACT_DIR = os.path.join(TARGET_DIR, "VOCdevkit_extracted")


def download(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    if not os.path.exists(TAR_PATH):
        download(VOC_URL, TAR_PATH)
    else:
        print(f"Tar file already exists: {TAR_PATH}")

    print("Extracting ...")
    with tarfile.open(TAR_PATH, "r") as tar:
        tar.extractall(EXTRACT_DIR, filter="data")

    voc_src = os.path.join(EXTRACT_DIR, "VOCdevkit", "VOC2012")
    voc_dst = os.path.join(TARGET_DIR, "VOCdevkit", "VOC2012")

    for sub in ["Annotations", "SegmentationClass", "SegmentationObject"]:
        src = os.path.join(voc_src, sub)
        dst = os.path.join(voc_dst, sub)
        if os.path.exists(src):
            if os.path.exists(dst):
                print(f"Skipping {sub} (already exists)")
                continue
            print(f"Copying {sub} ...")
            shutil.copytree(src, dst)
            print(f"  -> {dst}")

    print("Cleaning up extracted temp ...")
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    print("Done!")
    print(f"  Annotations:       {os.path.join(voc_dst, 'Annotations')}")
    print(f"  SegmentationClass: {os.path.join(voc_dst, 'SegmentationClass')}")


if __name__ == "__main__":
    main()
