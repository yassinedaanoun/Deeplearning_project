"""
Data lead module (Syrine) — Chest X-ray Pneumonia Detector.

Delivers the team contract locked at kickoff (see docs/interfaces.md):
    get_dataloaders(batch_size=32) -> (train_loader, val_loader, test_loader)

Expected on-disk layout (produced by scripts/download_dataset.py):
    data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

Input/label contract:
    - images resized to 224x224, 3 channels (grayscale channel repeated 3x
      since the pretrained ResNet expects 3-channel input)
    - labels are plain integers: 0 = NORMAL, 1 = PNEUMONIA

Note: the real Kaggle dataset's val/ folder only has 16 images total, which
is too small to validate on reliably. get_dataloaders() pools train+val and
re-splits them (stratified by class) by default -- see resplit_val below.
"""

import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGE_SIZE = 224
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "chest_xray"


def default_transform(train: bool = False) -> transforms.Compose:
    """Resize/normalize for eval; add light augmentation for training."""
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def _scan_split_dir(split_dir: Path) -> list:
    """Returns [(path, label), ...] for one split directory."""
    samples = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((img_path, label))
    return samples


class ChestXrayDataset(Dataset):
    """Loads NORMAL/PNEUMONIA images either from a split directory or an
    explicit list of (path, label) samples (used for the val resplit)."""

    def __init__(self, root_dir=None, transform=None, samples=None):
        self.transform = transform or default_transform(train=False)
        if samples is not None:
            self.samples = list(samples)
        else:
            self.root_dir = Path(root_dir)
            self.samples = _scan_split_dir(self.root_dir)
        if not self.samples:
            raise RuntimeError(f"No images found for {root_dir or 'given sample list'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    def class_counts(self) -> dict:
        counts = Counter(label for _, label in self.samples)
        return {name: counts.get(idx, 0) for name, idx in CLASS_TO_IDX.items()}


def get_class_counts(data_dir=DEFAULT_DATA_DIR) -> dict:
    """{split: {"NORMAL": n, "PNEUMONIA": n}} for each split present on disk."""
    data_dir = Path(data_dir)
    counts = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        samples = _scan_split_dir(split_dir)
        class_counts = Counter(label for _, label in samples)
        counts[split] = {name: class_counts.get(idx, 0) for name, idx in CLASS_TO_IDX.items()}
    return counts


def get_class_weights(data_dir=DEFAULT_DATA_DIR, split: str = "train") -> torch.Tensor:
    """Per-class weights (ordered by CLASS_TO_IDX) for a weighted loss function."""
    counts = get_class_counts(data_dir)[split]
    total = sum(counts.values())
    num_classes = len(CLASS_TO_IDX)
    weights = [total / (num_classes * max(counts[name], 1)) for name in CLASS_TO_IDX]
    return torch.tensor(weights, dtype=torch.float32)


def _make_weighted_sampler(dataset: ChestXrayDataset) -> WeightedRandomSampler:
    """Per-sample weights so each training batch sees a balanced class mix."""
    counts = dataset.class_counts()
    class_weight = {idx: 1.0 / max(counts[name], 1) for name, idx in CLASS_TO_IDX.items()}
    sample_weights = [class_weight[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _stratified_resplit(train_samples: list, val_samples: list, val_size: float, seed: int):
    """Pools train+val samples and re-splits them, stratified by class, so a
    tiny real val/ folder (e.g. 16 images) doesn't leave validation unreliable."""
    rng = random.Random(seed)
    by_class = {idx: [] for idx in CLASS_TO_IDX.values()}
    for path, label in train_samples + val_samples:
        by_class[label].append((path, label))

    new_train, new_val = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_val = round(len(items) * val_size)
        new_val.extend(items[:n_val])
        new_train.extend(items[n_val:])

    rng.shuffle(new_train)
    rng.shuffle(new_val)
    return new_train, new_val


def get_dataloaders(
    data_dir=DEFAULT_DATA_DIR,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
    resplit_val: bool = True,
    val_size: float = 0.2,
    seed: int = 42,
):
    """
    Team contract:
        get_dataloaders(batch_size=32) -> (train_loader, val_loader, test_loader)

    resplit_val=True (default) pools the official train/ and val/ folders and
    re-splits them (stratified by class) since the real dataset's val/ folder
    only has 16 images. Pass resplit_val=False to use the folders as-is.
    """
    data_dir = Path(data_dir)
    train_samples = _scan_split_dir(data_dir / "train")
    val_samples = _scan_split_dir(data_dir / "val")

    if resplit_val:
        train_samples, val_samples = _stratified_resplit(train_samples, val_samples, val_size, seed)

    train_ds = ChestXrayDataset(samples=train_samples, transform=default_transform(train=True))
    val_ds = ChestXrayDataset(samples=val_samples, transform=default_transform(train=False))
    test_ds = ChestXrayDataset(data_dir / "test", transform=default_transform(train=False))

    if use_weighted_sampler:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=_make_weighted_sampler(train_ds),
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    counts = get_class_counts()
    if not counts:
        print(f"No dataset found under {DEFAULT_DATA_DIR}.")
        print("Run scripts/download_dataset.py first.")
    else:
        for split, split_counts in counts.items():
            total = sum(split_counts.values())
            ratio = (
                split_counts["PNEUMONIA"] / split_counts["NORMAL"]
                if split_counts["NORMAL"]
                else float("inf")
            )
            print(f"{split}: {split_counts} total={total} pneumonia_to_normal_ratio={ratio:.2f}")
