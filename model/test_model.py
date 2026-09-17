"""
test_model.py — Independent sanity tests for model.py

Run this file directly to verify your model works correctly WITHOUT
needing any real X-ray data or teammates' code. Uses randomly generated
dummy tensors that match the agreed-upon input contract:
    (batch_size, 3, 224, 224)
"""

import torch
from model import build_model, count_trainable_params


def test_output_shape():
    """Model should output (batch_size, num_classes) logits."""
    model = build_model(num_classes=2)
    model.eval()

    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, 2), f"Expected shape (8, 2), got {output.shape}"
    print(f"[PASS] Output shape correct: {output.shape}")


def test_frozen_base_reduces_trainable_params():
    """Freezing the base should leave only the final layer trainable."""
    frozen_model = build_model(freeze_base=True)
    unfrozen_model = build_model(freeze_base=False)

    frozen_count = count_trainable_params(frozen_model)
    unfrozen_count = count_trainable_params(unfrozen_model)

    assert frozen_count < unfrozen_count, "Frozen model should have fewer trainable params"
    print(f"[PASS] Frozen trainable params: {frozen_count:,} < Unfrozen: {unfrozen_count:,}")


def test_unfreeze_layer4_variant():
    """Unfreezing layer4 should increase trainable params vs fully frozen."""
    fully_frozen = build_model(freeze_base=True, unfreeze_layer4=False)
    partially_unfrozen = build_model(freeze_base=True, unfreeze_layer4=True)

    fully_frozen_count = count_trainable_params(fully_frozen)
    partial_count = count_trainable_params(partially_unfrozen)

    assert partial_count > fully_frozen_count, "layer4 unfreeze should add trainable params"
    print(f"[PASS] layer4 unfrozen params: {partial_count:,} > fully frozen: {fully_frozen_count:,}")


def test_forward_and_backward_pass():
    """Simulate one training step end-to-end with fake data + fake labels."""
    model = build_model()
    model.train()

    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (4,))  # random 0/1 labels

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"[PASS] Forward + backward pass succeeded. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("Running independent model tests (no real data needed)...\n")
    test_output_shape()
    test_frozen_base_reduces_trainable_params()
    test_unfreeze_layer4_variant()
    test_forward_and_backward_pass()
    print("\nAll tests passed. model.py is ready to hand off to the Training lead.")