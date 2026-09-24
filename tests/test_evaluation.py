import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import compute_metrics, evaluate_model, plot_evaluation


def test_compute_metrics_confusion_and_roc_values():
    result = compute_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        pneumonia_scores=[0.1, 0.8, 0.4, 0.9],
    )

    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [1, 1]])
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["fpr"] is not None
    assert result["tpr"] is not None


def test_compute_metrics_zero_division_is_defined():
    result = compute_metrics([0, 1], [0, 0], [0.2, 0.4])

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_single_class_has_no_roc_data():
    result = compute_metrics([0, 0], [0, 1], [0.1, 0.8])

    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [0, 0]])
    assert result["roc_auc"] is None
    assert result["fpr"] is None
    assert result["tpr"] is None


def test_compute_metrics_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_metrics([0, 1], [0], [0.2, 0.8])


class ToyScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.grad_enabled_during_forward = None

    def forward(self, images):
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        score = images[:, 0] * self.scale
        return torch.stack((-score, score), dim=1)


def test_evaluate_model_uses_all_batches_without_training():
    images = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    model = ToyScoreModel()
    before = model.scale.detach().clone()
    model.train()

    result = evaluate_model(model, loader, device="cpu")

    assert result["confusion_matrix"].sum() == len(labels)
    assert result["accuracy"] == 1.0
    assert result["roc_auc"] == 1.0
    assert model.grad_enabled_during_forward is False
    assert model.training is False
    assert torch.equal(model.scale.detach(), before)


def test_evaluate_model_rejects_empty_loader():
    loader = DataLoader(
        TensorDataset(torch.empty((0, 1)), torch.empty((0,), dtype=torch.long))
    )

    with pytest.raises(ValueError, match="test_loader is empty"):
        evaluate_model(ToyScoreModel(), loader, device="cpu")


def test_plot_evaluation_saves_confusion_and_roc(tmp_path):
    result = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])

    paths = plot_evaluation(result, tmp_path)

    assert paths["confusion_matrix"].is_file()
    assert paths["roc_curve"].is_file()


def test_plot_evaluation_skips_roc_for_single_class(tmp_path):
    result = compute_metrics([0, 0], [0, 1], [0.1, 0.8])

    paths = plot_evaluation(result, tmp_path)

    assert paths == {"confusion_matrix": tmp_path / "confusion_matrix.png"}
    assert paths["confusion_matrix"].is_file()

