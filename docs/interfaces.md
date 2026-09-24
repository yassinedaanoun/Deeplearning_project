# Shared interfaces (locked at Step 0 kickoff)

This is the contract everyone builds against so all five roles can work in
parallel and plug together without surprises at integration time. Don't
change a signature here without flagging it to the whole team.

## Input format

- Image size: **224 × 224**
- Channels: **3** (X-rays are grayscale, but the pretrained ResNet expects
  3 channels — the single channel is repeated 3×)
- Tensor shape convention: `(batch_size, 3, 224, 224)`

## Label format

- `0` = Normal, `1` = Pneumonia
- Labels are **plain integers**, not one-hot — works directly with
  `CrossEntropyLoss`

## Function signatures

```python
# Data lead delivers (data.py):
get_dataloaders(batch_size=32) -> (train_loader, val_loader, test_loader)

# Preprocessing lead delivers (preprocessing.py):
get_transforms(split="train") -> torchvision.transforms.Compose
# split in {"train", "val", "test"} — "train" includes augmentation,
# "val"/"test" are deterministic (resize + normalize only); data.py uses
# this interface for all three splits.

# Model lead delivers (model/model.py, exported by model/__init__.py):
build_model(num_classes=2) -> model

# Training lead delivers (train.py):
train_model(model, train_loader, val_loader, criterion, optimizer,
            num_epochs=10) -> (trained_model, history)

# Evaluation lead delivers (evaluate.py):
compute_metrics(y_true, y_pred, pneumonia_scores) -> dict
evaluate_model(model, test_loader, device=None) -> dict
# Keys: confusion_matrix, accuracy, precision, recall, f1, roc_auc, fpr, tpr.
# Positive class is 1 = PNEUMONIA; ROC fields are None for one-class labels.

# Demo lead delivers (demo.py):
predict(model, image, device=None) -> (label, confidence)
# label is NORMAL or PNEUMONIA; confidence is its softmax score in [0, 1].
```

## Repo structure

```
project/
  data.py          # Data lead (Syrine)
  preprocessing.py # Preprocessing lead (Ayoub)
  model/           # Model lead (Omar): model.py + public exports
  train.py         # Training lead (Romaric)
  evaluate.py       \
  demo.py            > Evaluation + Demo lead (Yassine)
  main.py          # glues it together at integration points
```

## Independent task split — how each role tests without blocking on anyone else

| # | Role | Owner | Tests independently using |
|---|------|-------|----------------------------|
| 1 | Data lead | Syrine | The real downloaded dataset — fully independent, needs nothing |
| 2 | Preprocessing lead | Ayoub | A handful of sample images downloaded separately — doesn't wait for the Data lead's finished pipeline, just needs *some* images to test resize/normalize/augment on |
| 3 | Model lead | Omar | Dummy tensors `torch.randn(8, 3, 224, 224)` — a model only cares about shape, not where the numbers came from |
| 4 | Training lead | Romaric | A tiny public dataset already in `torchvision.datasets` (e.g. a small CIFAR-10 subset), or dummy tensors + dummy labels — proves the loop mechanics (forward, loss, backward, optimizer step, logging) without needing real pneumonia data or the final model |
| 5 | Evaluation + Demo lead | Yassine | Fake label/score arrays for metrics; a dummy classifier for prediction and UI construction |

Everyone builds to the shapes/signatures above, so at the integration
checkpoints these all plug together without anyone having waited on
anyone else.

## Integration checkpoints

- **End of Day 2:** Data + Preprocessing + Model + Training loop plugged
  together for the first real run. `data.py` now uses
  `preprocessing.get_transforms()` for each split.
- **End of Day 4:** Trained model plugged into Evaluation + Demo

## Status

- [x] Data lead — `data.py` (`get_dataloaders`) — done, see branch
      `step1/data-lead`
- [x] Preprocessing lead — `preprocessing.py` (`get_transforms`)
- [x] Model lead — `model/model.py` (`build_model`)
- [x] Training lead — `train.py` (`train_model`)
- [x] Evaluation lead — `evaluate.py` (`evaluate_model`)
- [x] Demo lead — `demo.py` (`predict`)

The Yassine modules are independently testable with synthetic inputs. Their
real-data integration requires Romaric's ResNet checkpoint at
`models/best_model.pth`.

### Data and preprocessing integration

`data.py` uses `preprocessing.get_transforms(split)` for train, validation,
and test images. The shared transform function returns 3-channel,
224x224 ImageNet-normalized tensors; only the training split is augmented.
