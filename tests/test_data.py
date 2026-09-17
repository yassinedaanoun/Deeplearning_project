"""
Tests for data.py — uses a small synthetic dataset shaped exactly like the
real chest_xray/ folder (train/val/test x NORMAL/PNEUMONIA) so the whole
pipeline is exercised without needing the actual ~1.2GB Kaggle download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as data_module  # noqa: E402


def _make_fake_image(path: Path, size=(300, 260)):
    """Grayscale-looking image, like a real chest X-ray export."""
    arr = np.random.randint(0, 255, size=(size[1], size[0]), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def fake_dataset(tmp_path):
    """
    Builds:
        train: NORMAL=12, PNEUMONIA=36   (3:1 imbalance, like the real set)
        val:   NORMAL=2,  PNEUMONIA=2    (tiny on purpose -> triggers resplit)
        test:  NORMAL=6,  PNEUMONIA=10
    """
    root = tmp_path / "chest_xray"
    layout = {
        "train": {"NORMAL": 12, "PNEUMONIA": 36},
        "val": {"NORMAL": 2, "PNEUMONIA": 2},
        "test": {"NORMAL": 6, "PNEUMONIA": 10},
    }
    for split, classes in layout.items():
        for class_name, n in classes.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                _make_fake_image(class_dir / f"img_{i}.jpeg")
    return root, layout


def test_get_class_counts(fake_dataset):
    root, layout = fake_dataset
    counts = data_module.get_class_counts(root)
    for split, classes in layout.items():
        for class_name, n in classes.items():
            assert counts[split][class_name] == n


def test_class_weights_reflect_imbalance(fake_dataset):
    root, _ = fake_dataset
    weights = data_module.get_class_weights(root, split="train")
    # PNEUMONIA (idx 1) is the majority class -> should get the *smaller* weight
    assert weights[data_module.CLASS_TO_IDX["PNEUMONIA"]] < weights[data_module.CLASS_TO_IDX["NORMAL"]]


def test_get_dataloaders_contract_shapes(fake_dataset):
    root, layout = fake_dataset
    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        batch_size=4, data_dir=root, num_workers=0
    )

    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert set(labels.tolist()).issubset({0, 1})

    # test split must be untouched by the resplit logic
    n_test = layout["test"]["NORMAL"] + layout["test"]["PNEUMONIA"]
    assert len(test_loader.dataset) == n_test


def test_val_resplit_triggers_on_tiny_val(fake_dataset):
    root, layout = fake_dataset
    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        batch_size=4, data_dir=root, num_workers=0, resplit_val=True, val_size=0.2, seed=0
    )
    n_train_orig = layout["train"]["NORMAL"] + layout["train"]["PNEUMONIA"]
    n_val_orig = layout["val"]["NORMAL"] + layout["val"]["PNEUMONIA"]
    pooled = n_train_orig + n_val_orig

    # val grew well past the original 4 images because we pooled + resplit
    assert len(val_loader.dataset) > n_val_orig
    assert len(train_loader.dataset) + len(val_loader.dataset) == pooled


def test_val_resplit_skipped_when_disabled(fake_dataset):
    root, layout = fake_dataset
    _, val_loader, _ = data_module.get_dataloaders(
        batch_size=4, data_dir=root, num_workers=0, resplit_val=False
    )
    n_val_orig = layout["val"]["NORMAL"] + layout["val"]["PNEUMONIA"]
    assert len(val_loader.dataset) == n_val_orig


def test_labels_are_plain_ints_not_onehot(fake_dataset):
    root, _ = fake_dataset
    train_loader, _, _ = data_module.get_dataloaders(batch_size=4, data_dir=root, num_workers=0)
    _, labels = next(iter(train_loader))
    assert labels.ndim == 1  # not one-hot
