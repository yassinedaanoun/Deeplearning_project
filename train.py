"""Moteur d'entraînement PyTorch pour le projet de classification de pneumonie.

Contrat attendu par ce module :
- model(images) -> renvoie des logits de forme [taille_du_batch, 2]
- labels ont la forme [taille_du_batch] et contiennent uniquement les valeurs 0 ou 1
- train_loader et val_loader renvoient des couples (images, labels)

"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def get_default_device() -> torch.device:
    """Retourner CUDA, puis Apple MPS, et sinon le processeur (CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Retourner le nombre de prédictions correctes dans un lot."""
    predictions = logits.argmax(dim=1)
    return int((predictions == labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entraîner ``model`` pendant exactement une époque.

    Retourne :
        (perte_moyenne, précision)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1) Supprimer les gradients laissés par le lot précédent.
        optimizer.zero_grad()

        # 2) Propagation avant : obtenir les scores bruts des classes (logits).
        logits = model(images)

        # 3) Comparer les prédictions aux étiquettes de référence.
        loss = criterion(logits, labels)

        # 4) Rétropropagation : calculer les gradients de la perte.
        loss.backward()

        # 5) Mettre à jour tous les paramètres entraînables gérés par l'optimiseur.
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += _batch_accuracy(logits, labels)
        total += batch_size

    if total == 0:
        raise ValueError("train_loader is empty; at least one sample is required.")

    average_loss = running_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Évaluer ``model`` pendant une époque complète de validation, sans l'entraîner.

    Retourne :
        (perte_moyenne, précision)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Aucun graphe de gradients n'est nécessaire pendant la validation.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += _batch_accuracy(logits, labels)
            total += batch_size

    if total == 0:
        raise ValueError("val_loader is empty; at least one sample is required.")

    average_loss = running_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device | None = None,
    checkpoint_path: str | Path = "best_model.pth",
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Entraîner un modèle et restaurer les poids de sa meilleure époque de validation.

    La meilleure époque est sélectionnée selon la précision de validation.

    Arguments :
        model: Modèle de classification PyTorch compatible.
        train_loader: DataLoader utilisé pour l'apprentissage basé sur les gradients.
        val_loader: DataLoader utilisé uniquement pour la validation.
        criterion: Fonction de perte, par exemple nn.CrossEntropyLoss(...).
        optimizer: Optimiseur créé en dehors de cette fonction.
        num_epochs: Nombre de passages complets sur les données d'entraînement.
        device: Appareil sur lequel exécuter le calcul. Si None, il est sélectionné automatiquement.
        checkpoint_path: Fichier dans lequel le meilleur point de sauvegarde est enregistré.
        verbose: Afficher une ligne récapitulative par époque si True.

    Retourne :
        trained_model: ``model`` restauré avec ses meilleurs poids de validation.
        history: Courbes de perte/précision et métadonnées de la meilleure époque.
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0.")

    if device is None:
        device = get_default_device()

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    history: dict[str, Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": None,
        "best_val_accuracy": None,
        "device": str(device),
    }

    best_val_accuracy = float("-inf")
    best_model_state: dict[str, torch.Tensor] | None = None

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            history["best_epoch"] = epoch + 1
            history["best_val_accuracy"] = val_accuracy

            # deepcopy est important : model.state_dict() seul continuerait à
            # référencer des tenseurs qui changent au cours des époques suivantes.
            best_model_state = deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "history": deepcopy(history),
                },
                checkpoint_path,
            )

        if verbose:
            marker = "  <-- best" if is_best else ""
            print(
                f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy * 100:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy * 100:6.2f}%"
                f"{marker}"
            )

    # Restaurer les meilleurs poids de validation avant de retourner le modèle.
    if best_model_state is None:
        raise RuntimeError("No best model state was produced during training.")

    model.load_state_dict(best_model_state)

    return model, history
