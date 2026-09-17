"""
Generates a tiny synthetic dataset with the same folder layout as the real
Kaggle data, so anyone (data lead included) can develop and test data.py,
model.py, train.py, etc. without waiting on the real ~5,863-image download.

Usage:
    python scripts/make_dummy_dataset.py [--out data/dummy_chest_xray] [--per-class 6]
"""

import argparse
import random
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def make_dataset(out_dir: Path, per_class: int, image_size: int = 256) -> None:
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = out_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                color = random.randint(0, 255)
                img = Image.new("L", (image_size, image_size), color=color)
                img.save(class_dir / f"{class_name.lower()}_{i}.jpeg")
    print(f"Dummy dataset written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy_chest_xray")
    parser.add_argument("--per-class", type=int, default=6)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    make_dataset(root / args.out, args.per_class)
