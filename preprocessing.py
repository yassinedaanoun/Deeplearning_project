"""Image transforms shared by the data loader and inference code."""

from torchvision import transforms


def get_transforms(split: str = "train") -> transforms.Compose:
    """Return the image transforms for training, validation, or test."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")

    ops = [transforms.Grayscale(num_output_channels=3)]
    if split == "train":
        ops += [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops.append(transforms.Resize((224, 224)))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(ops)


def get_train_transforms() -> transforms.Compose:
    """Backward-compatible training transform helper."""
    return get_transforms("train")


def get_val_transforms() -> transforms.Compose:
    """Backward-compatible validation/test transform helper."""
    return get_transforms("val")
