import pytest
import torch
from torch import nn

import model.model as model_module
from model import load_trained_model


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward(self, images):
        return self.head(images)


def test_load_trained_model_restores_checkpoint_without_pretrained_download(
    tmp_path, monkeypatch
):
    expected_model = TinyClassifier()
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": expected_model.state_dict()}, checkpoint_path)
    monkeypatch.setattr(
        model_module, "build_model", lambda pretrained=False: TinyClassifier()
    )

    loaded_model = load_trained_model(checkpoint_path, torch.device("cpu"))

    assert loaded_model.training is False
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            expected_model.state_dict().values(), loaded_model.state_dict().values()
        )
    )


def test_load_trained_model_reports_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_trained_model(tmp_path / "missing.pth", torch.device("cpu"))


def test_load_trained_model_reports_missing_state_dict(tmp_path):
    checkpoint_path = tmp_path / "invalid.pth"
    torch.save({"epoch": 2}, checkpoint_path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_trained_model(checkpoint_path, torch.device("cpu"))


def test_load_trained_model_reports_incompatible_weights(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "incompatible.pth"
    torch.save({"model_state_dict": {"unexpected": torch.tensor(1)}}, checkpoint_path)
    monkeypatch.setattr(
        model_module, "build_model", lambda pretrained=False: TinyClassifier()
    )

    with pytest.raises(ValueError, match="incompatible with the ResNet18 classifier"):
        load_trained_model(checkpoint_path, torch.device("cpu"))
