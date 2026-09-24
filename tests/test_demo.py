import math

import pytest
import torch
from PIL import Image
from torch import nn

from demo import DISCLAIMER, build_demo, predict


class DummyClassifier(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.last_input_shape = None
        self.grad_enabled_during_forward = None

    def forward(self, images):
        self.last_input_shape = tuple(images.shape)
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return self.logits.unsqueeze(0).expand(images.shape[0], -1)


@pytest.mark.parametrize(
    ("image", "logits", "expected_label"),
    [
        (Image.new("L", (80, 60), color=120), [3.0, 0.0], "NORMAL"),
        (Image.new("RGB", (60, 80), color=(120, 120, 120)), [0.0, 3.0], "PNEUMONIA"),
    ],
)
def test_predict_accepts_pil_images_and_returns_class_score(image, logits, expected_label):
    model = DummyClassifier(logits)

    label, confidence = predict(model, image, device="cpu")

    assert label == expected_label
    assert math.isfinite(confidence)
    assert 0.0 <= confidence <= 1.0
    assert model.last_input_shape == (1, 3, 224, 224)
    assert model.grad_enabled_during_forward is False
    assert model.training is False


def test_predict_rejects_missing_or_non_pil_image():
    model = DummyClassifier([1.0, 0.0])

    with pytest.raises(ValueError, match="Upload an X-ray image"):
        predict(model, None, device="cpu")
    with pytest.raises(TypeError, match="PIL image"):
        predict(model, "xray.png", device="cpu")


def test_build_demo_constructs_interface_without_launching_server():
    demo = build_demo(DummyClassifier([1.0, 0.0]), device="cpu")

    assert demo.__class__.__name__ == "Interface"
    assert DISCLAIMER in demo.description

