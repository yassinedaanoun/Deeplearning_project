"""Test d'acceptation autonome pour train.py.
L'objectif est de démontrer que la mécanique d'entraînement est correcte et indépendante.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    get_default_device,
    train_model,
    train_one_epoch,
    validate_one_epoch,
)


def make_toy_loaders(batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    """Créer un jeu de données déterministe et simple à deux classes, semblable à des images.

    Les images de la classe 0 sont centrées autour de -1 et celles de la classe 1
    autour de +1, afin qu'un petit classifieur puisse apprendre rapidement la
    séparation. Les images utilisent le même nombre de canaux que celui attendu
    par ResNet (3), mais une taille spatiale réduite pour accélérer le test.
    """
    generator = torch.Generator().manual_seed(1234)

    n_train_per_class = 64
    n_val_per_class = 24
    shape = (3, 8, 8)

    train_0 = torch.randn(n_train_per_class, *shape, generator=generator) * 0.25 - 1.0
    train_1 = torch.randn(n_train_per_class, *shape, generator=generator) * 0.25 + 1.0
    val_0 = torch.randn(n_val_per_class, *shape, generator=generator) * 0.25 - 1.0
    val_1 = torch.randn(n_val_per_class, *shape, generator=generator) * 0.25 + 1.0

    x_train = torch.cat([train_0, train_1], dim=0)
    y_train = torch.cat(
        [
            torch.zeros(n_train_per_class, dtype=torch.long),
            torch.ones(n_train_per_class, dtype=torch.long),
        ]
    )

    x_val = torch.cat([val_0, val_1], dim=0)
    y_val = torch.cat(
        [
            torch.zeros(n_val_per_class, dtype=torch.long),
            torch.ones(n_val_per_class, dtype=torch.long),
        ]
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def build_tiny_external_model() -> nn.Module:
    """Un modèle défini volontairement en dehors de train.py."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )


def clone_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def any_parameter_changed(
    before: list[torch.Tensor], model: nn.Module
) -> bool:
    after = [parameter.detach().cpu() for parameter in model.parameters()]
    return any(not torch.equal(old, new) for old, new in zip(before, after))


def main() -> None:
    torch.manual_seed(1234)

    device = get_default_device()
    print(f"Device selected: {device}")

    train_loader, val_loader = make_toy_loaders()
    model = build_tiny_external_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Test d'acceptation 1 : une époque d'entraînement autonome met à jour les paramètres.
    parameters_before = clone_parameters(model)
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    assert train_loss >= 0.0
    assert 0.0 <= train_acc <= 1.0
    assert any_parameter_changed(parameters_before, model), (
        "optimizer.step() did not modify any model parameter"
    )

    # Test d'acceptation 2 : la validation ne doit PAS mettre à jour les paramètres.
    parameters_before_validation = clone_parameters(model)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    parameters_after_validation = clone_parameters(model)
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before_validation, parameters_after_validation)
    ), "Validation changed model parameters"

    # Utiliser un nouveau modèle externe pour le test complet de train_model.
    model = build_tiny_external_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    output_dir = Path(__file__).resolve().parent / "artifacts"
    checkpoint_path = output_dir / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=4,
        device=device,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    # Test d'acceptation 3 : l'historique attendu existe pour chaque époque.
    for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]:
        assert key in history
        assert len(history[key]) == 4

    assert history["best_epoch"] in {1, 2, 3, 4}
    assert 0.0 <= history["best_val_accuracy"] <= 1.0

    # Test d'acceptation 4 : le point de sauvegarde existe et contient les métadonnées d'intégration.
    assert checkpoint_path.exists(), "best_model.pth was not created"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in [
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "val_accuracy",
        "history",
    ]:
        assert key in checkpoint, f"Missing checkpoint key: {key}"

    assert checkpoint["epoch"] == history["best_epoch"]

    # Test d'acceptation 5 : le modèle retourné correspond aux meilleurs poids sauvegardés.
    returned_state = {
        key: value.detach().cpu() for key, value in trained_model.state_dict().items()
    }
    saved_state = checkpoint["model_state_dict"]
    assert returned_state.keys() == saved_state.keys()
    assert all(
        torch.equal(returned_state[key], saved_state[key]) for key in returned_state
    ), "Returned model is not restored to the best checkpoint weights"

    # Test d'acceptation 6 : le problème jouet devrait être appris facilement.
    final_val_accuracy = history["val_accuracy"][-1]
    assert max(history["val_accuracy"]) >= 0.95, (
        "Toy problem was not learned; training mechanics may be incorrect"
    )

    print("\nALL ACCEPTANCE TESTS PASSED")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation accuracy: {history['best_val_accuracy'] * 100:.2f}%")
    print(f"Final validation accuracy: {final_val_accuracy * 100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
