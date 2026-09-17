# Model — ResNet18 for Chest X-ray Classification

## What this does
`build_model()` returns a ResNet18 (pretrained on ImageNet) adapted to classify
chest X-rays as **Normal (0)** or **Pneumonia (1)**.

## Contract
```python
from model import build_model

model = build_model(num_classes=2, freeze_base=True, unfreeze_layer4=False)
```
- **Input:** tensor of shape `(batch_size, 3, 224, 224)`
- **Output:** raw logits of shape `(batch_size, 2)` — no softmax applied,
  use `CrossEntropyLoss` directly or apply `softmax`/`argmax` yourself for predictions

## Two training options

### 1. Frozen base (default) — start here
```python
model = build_model()  # freeze_base=True, unfreeze_layer4=False
```
Only the final classifier layer trains (**1,026 parameters**). Everything else
keeps its original ImageNet weights.

- ✅ Fast to train (few params to update)
- ✅ Less risk of overfitting on our small dataset (~5,800 images)
- ✅ Good first attempt — use this for the initial training run

### 2. Partially unfrozen (layer4) — use if accuracy is too low
```python
model = build_model(unfreeze_layer4=True)
```
Also unfreezes ResNet's last conv block, `layer4` (**8,394,754 trainable
parameters** in addition to the classifier).

- Lets the model adapt deeper features specifically to X-ray images instead
  of relying purely on generic ImageNet features
- ✅ Try this **only if** the frozen version underperforms — more params
  training means more risk of overfitting on a small dataset, and longer
  training time
- Recommended: lower the learning rate slightly when using this option
  (e.g., `1e-5` instead of `1e-4`) since more of the network is now learning

## Recommendation for this week
Start with the **default frozen version**. Only switch to `unfreeze_layer4=True`
if validation accuracy plateaus too low (e.g., below ~85%) after tuning the
frozen version's training (learning rate, epochs, augmentation).

## Testing
All logic is verified independently of real data — see `test_model.py`.
Run it after pulling to confirm your environment works correctly:
```bash
python3 model/test_model.py
```
Expected: 4x `[PASS]`, ending in a successful forward+backward pass.