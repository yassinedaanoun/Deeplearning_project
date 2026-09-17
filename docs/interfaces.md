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
# "val"/"test" are deterministic (resize + normalize only). Must produce
# the same 224x224 / 3-channel / ImageNet-normalized output data.py
# already relies on, so it's a drop-in swap at integration (see note below).

# Model lead delivers (model.py):
build_model(num_classes=2) -> model

# Training lead delivers (train.py):
train_model(model, train_loader, val_loader, epochs=10) -> (trained_model, history)

# Evaluation lead delivers (evaluate.py):
evaluate_model(model, test_loader) -> (confusion_matrix, precision, recall, f1)

# Demo lead delivers (demo.py):
predict(model, image) -> (label, confidence)
```

## Repo structure

```
project/
  data.py          # Data lead (Syrine)
  preprocessing.py # Preprocessing lead (Ayoub)
  model.py         # Model lead (Omar)
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
| 5 | Evaluation + Demo lead | Yassine | Fake prediction arrays (`np.random.randint(0,2,100)` vs fake ground truth) for the metrics code; a dummy model that returns random predictions for the demo UI — tests the *code*, not real results |

Everyone builds to the shapes/signatures above, so at the integration
checkpoints these all plug together without anyone having waited on
anyone else.

## Integration checkpoints

- **End of Day 2:** Data + Preprocessing + Model + Training loop plugged
  together for the first real run. This is also when `data.py` should
  swap its current inline transforms for `preprocessing.get_transforms()`
  (see note below) once Ayoub's module lands.
- **End of Day 4:** Trained model plugged into Evaluation + Demo

## Status

- [x] Data lead — `data.py` (`get_dataloaders`) — done, see branch
      `step1/data-lead`
- [ ] Preprocessing lead — `preprocessing.py` (`get_transforms`)
- [ ] Model lead — `model.py` (`build_model`)
- [ ] Training lead — `train.py` (`train_model`)
- [ ] Evaluation lead — `evaluate.py` (`evaluate_model`)
- [ ] Demo lead — `demo.py` (`predict`)

### Note on `data.py` and `preprocessing.py`

Step 0 left the transform pipeline allowed to live "merged into data.py
or its own file." `data.py` currently ships its own internal
`_train_transform()` / `_eval_transform()` (224x224 resize, ImageNet
normalize, train-only flip/rotation) so the Data lead's deliverable is
fully self-contained and testable today. Now that Preprocessing is a
separate role, the integration step is: once `preprocessing.py` exposes
`get_transforms(split)` matching the contract above, swap the two
internal helper calls in `data.py`'s `ChestXrayDataset` construction for
`preprocessing.get_transforms(split)` — everything downstream
(shapes, dtypes, the resplit logic) is unaffected either way.
