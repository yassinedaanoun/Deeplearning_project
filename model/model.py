"""
model.py — Model Lead deliverable

Builds a ResNet18 model (pretrained on ImageNet) adapted for
binary chest X-ray classification: Normal (0) vs Pneumonia (1).

Contract with the rest of the team:
    build_model(num_classes=2, freeze_base=True, unfreeze_layer4=False) -> nn.Module
    Input shape expected by the model: (batch_size, 3, 224, 224)
    Output shape: (batch_size, num_classes)  -- raw logits, no softmax applied
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


def build_model(
    num_classes: int = 2,
    freeze_base: bool = True,
    unfreeze_layer4: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet18 model adapted for our classification task.

    Args:
        num_classes: number of output classes (2 for Normal/Pneumonia)
        freeze_base: if True, freezes all pretrained layers except the final
                     classifier layer (fc). This is the default / safest option.
        unfreeze_layer4: if True (and freeze_base is True), also unfreezes the
                          last ResNet block (layer4) so it can adapt slightly
                          deeper features to X-ray images. Optional stretch
                          goal — try this if the fully-frozen version
                          underperforms.
        pretrained: load ImageNet weights. Set to False when loading a complete
                    project checkpoint, which already contains all model weights.

    Returns:
        A torchvision ResNet18 model with a new final layer, ready for training.
    """
    # A complete project checkpoint does not need an ImageNet download first.
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all layers by default so only the new classifier head trains
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

        # Optional: unfreeze the last conv block for slightly deeper fine-tuning
        if unfreeze_layer4:
            for param in model.layer4.parameters():
                param.requires_grad = True

    # Replace the final fully-connected layer.
    # ResNet18's default fc outputs 1000 classes (ImageNet) — we only need num_classes.
    # NOTE: newly created layers have requires_grad=True by default, so this
    # layer will always train regardless of freeze_base.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load a train_model checkpoint onto ``device`` and return it in eval mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except Exception as exc:
        raise ValueError(f"Could not read model checkpoint: {checkpoint_path}") from exc

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint {checkpoint_path}: expected a 'model_state_dict' entry."
        )

    model = build_model(pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint weights are incompatible with the ResNet18 classifier: "
            f"{checkpoint_path}"
        ) from exc

    return model.to(device).eval()


def count_trainable_params(model: nn.Module) -> int:
    """Utility: count how many parameters will actually be updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    model = build_model()
    print(model)
    print(f"\nTrainable parameters: {count_trainable_params(model):,}")
