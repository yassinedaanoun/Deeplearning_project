"""
Downloads and organizes the "Chest X-Ray Images (Pneumonia)" Kaggle dataset
into the layout data.py expects: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

Requires a free Kaggle account and API credentials (see README.md for the
full walkthrough):
    1. Create a token at https://www.kaggle.com/settings -> "Create New Token"
    2. Save it as the only line in ~/.kaggle/access_token
       (Windows: %USERPROFILE%\.kaggle\access_token)

Usage:
    pip install kagglehub
    python scripts/download_dataset.py
"""

import shutil
from pathlib import Path

import kagglehub

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "chest_xray"


def main():
    if TARGET_DIR.exists():
        print(f"{TARGET_DIR} already exists — delete it first to re-download.")
        return

    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Downloaded to {download_path}")

    # The Kaggle archive nests the real split folders one level down,
    # e.g. <download_path>/chest_xray/{train,val,test}.
    source = download_path
    nested = download_path / "chest_xray"
    if nested.is_dir():
        source = nested

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, TARGET_DIR)
    print(f"Dataset ready at {TARGET_DIR}")


if __name__ == "__main__":
    main()
