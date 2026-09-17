# Romaric — Training Lead

This folder contains the independent PyTorch training engine for the Chest X-ray Pneumonia Detector project.

## Contract

- `train_loader` / `val_loader` yield `(images, labels)`
- final project images: `[B, 3, 224, 224]`
- labels: `[B]`, where `0 = NORMAL`, `1 = PNEUMONIA`
- model output: `[B, 2]` logits
- recommended loss for the project: `nn.CrossEntropyLoss(...)`

The training code does **not** create the dataset and does **not** create ResNet18.
Those objects are injected from the other team members.

## Run the independent acceptance test

```bash
python3 test_training.py
```

The test uses a tiny synthetic binary image dataset, so it requires no Kaggle download and no network access.

## Final integration example

```python
import torch
from torch import nn
from train import get_default_device, train_model

# Fourni par les autres membres de l'équipe :
# model = build_model()
# train_loader, val_loader, test_loader = get_dataloaders()

device = get_default_device()
criterion = nn.CrossEntropyLoss()  # can later receive class weights
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
)

trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=10,
    device=device,
    checkpoint_path="models/best_model.pth",
)
```
