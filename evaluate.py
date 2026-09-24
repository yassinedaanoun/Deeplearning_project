"""Metrics and plots for evaluating the pneumonia classifier on the test split."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def compute_metrics(
    y_true, y_pred, pneumonia_scores
) -> dict[str, Any]:
    """Compute binary metrics; class 1 (PNEUMONIA) is the positive class.

    ``pneumonia_scores`` must contain the model probability for class 1 for
    each sample. ROC/AUC are omitted when ``y_true`` contains only one class.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    pneumonia_scores = np.asarray(pneumonia_scores, dtype=np.float64).reshape(-1)

    if not y_true.size:
        raise ValueError("At least one test sample is required to compute metrics.")
    if not (y_true.size == y_pred.size == pneumonia_scores.size):
        raise ValueError("y_true, y_pred, and pneumonia_scores must have equal lengths.")
    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
        raise ValueError("Labels and predictions must use 0=NORMAL and 1=PNEUMONIA.")
    if not np.isfinite(pneumonia_scores).all():
        raise ValueError("pneumonia_scores must contain only finite values.")
    if ((pneumonia_scores < 0) | (pneumonia_scores > 1)).any():
        raise ValueError("pneumonia_scores must be probabilities between 0 and 1.")

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    has_both_classes = np.unique(y_true).size == 2

    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, pneumonia_scores, pos_label=1)
        roc_auc = float(roc_auc_score(y_true, pneumonia_scores))
    else:
        fpr = tpr = None
        roc_auc = None

    return {
        "confusion_matrix": matrix,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run inference on ``test_loader`` and return binary metrics and ROC data."""
    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_pneumonia_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("The model must return logits with shape [batch_size, 2].")

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_pneumonia_scores.extend(probabilities[:, 1].detach().cpu().tolist())

    if not all_labels:
        raise ValueError("test_loader is empty; at least one sample is required.")

    return compute_metrics(all_labels, all_predictions, all_pneumonia_scores)


def plot_evaluation(
    result: dict[str, Any], output_dir: str | Path = "artifacts/evaluation"
) -> dict[str, Path]:
    """Save a labeled confusion matrix and, when defined, the ROC curve."""
    import matplotlib

    # This module saves files only; select a non-interactive backend so plotting
    # also works in headless environments and test runs.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = output_dir / "confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"], display_labels=CLASS_NAMES
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Matrice de confusion — test")
    axis.set_xlabel("Prédiction")
    axis.set_ylabel("Vérité terrain")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    paths = {"confusion_matrix": confusion_path}
    if result["roc_auc"] is not None:
        roc_path = output_dir / "roc_curve.png"
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(
            result["fpr"],
            result["tpr"],
            label=f"ROC (AUC = {result['roc_auc']:.3f})",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
        axis.set(
            title="Courbe ROC — classe PNEUMONIA",
            xlabel="Taux de faux positifs",
            ylabel="Rappel (taux de vrais positifs)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(roc_path, dpi=150)
        plt.close(figure)
        paths["roc_curve"] = roc_path

    return paths


def main() -> None:
    from data import get_dataloaders
    from model import load_trained_model
    from train import get_default_device

    parser = argparse.ArgumentParser(description="Evaluate the chest X-ray classifier.")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_default_device()
    model = load_trained_model(args.checkpoint, device)
    # A single-process loader also works in restricted macOS environments where
    # PyTorch's shared-memory manager cannot start worker processes.
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=0)
    result = evaluate_model(model, test_loader, device=device)
    paths = plot_evaluation(result, args.output_dir)

    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision (PNEUMONIA): {result['precision']:.4f}")
    print(f"Recall (PNEUMONIA):    {result['recall']:.4f}")
    print(f"F1 (PNEUMONIA):        {result['f1']:.4f}")
    if result["roc_auc"] is None:
        print("ROC AUC: unavailable (test labels contain only one class)")
    else:
        print(f"ROC AUC: {result['roc_auc']:.4f}")
    print(f"Confusion matrix: {result['confusion_matrix'].tolist()}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
