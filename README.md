# Deeplearning_project

Chest X-ray Pneumonia Detector — binary classification (NORMAL vs PNEUMONIA)
via transfer learning (ResNet18) on the Kaggle "Chest X-Ray Images (Pneumonia)"
dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline (data lead)

`data.py` delivers the team contract:

```python
get_dataloaders(batch_size=32) -> (train_loader, val_loader, test_loader)
```

- Images are resized to 224x224 and expanded to 3 channels (grayscale
  repeated) to match the pretrained ResNet input.
- Labels are plain integers: `0 = NORMAL`, `1 = PNEUMONIA`.
- The training loader uses a `WeightedRandomSampler` to counter the ~3:1
  pneumonia/normal class imbalance in the dataset.

### Generating the real dataset

You need a free Kaggle account and an API token — this only has to be done
once per machine.

**1. Get a Kaggle API token**

1. Go to https://www.kaggle.com/settings
2. Scroll to the **API** section and click **Create New Token**
   (this downloads/shows a token starting with `KGAT_`)
3. Copy it somewhere safe for a moment — you won't be able to see it again,
   only regenerate a new one. Treat it like a password: never commit it,
   paste it in a chat, or check it into git.

**2. Save the token locally**

Windows (PowerShell):
```powershell
mkdir $env:USERPROFILE\.kaggle -Force
notepad $env:USERPROFILE\.kaggle\access_token
```
Paste the token as the only line in the file, save, and close Notepad.

macOS/Linux:
```bash
mkdir -p ~/.kaggle
echo "PASTE_YOUR_TOKEN_HERE" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**3. Install dependencies (if you haven't already)**

```bash
pip install -r requirements.txt
```
(On Windows use `python -m pip install -r requirements.txt` if `pip` isn't
directly on your PATH.)

**4. Download and organize the dataset**

```bash
python scripts/download_dataset.py
```

This pulls the ~1.2GB "Chest X-Ray Images (Pneumonia)" dataset via
`kagglehub` and lays it out at `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`
— it can take a few minutes depending on your connection.

**5. Verify it worked / see the class balance**

```bash
python data.py
```

This prints the per-split class counts (see the real output below). If you
get `No dataset found under .../data/chest_xray`, the download step above
either didn't run or didn't finish — re-run step 4.

**6. Rotate your token when you're done sharing/pairing on this**

If your token was ever pasted somewhere outside your own machine (chat,
screen share, etc.), regenerate it at https://www.kaggle.com/settings
("Create New Token" again invalidates the old one).

To develop/test without the real download, generate a tiny synthetic dataset
with the same folder layout:

```bash
python scripts/make_dummy_dataset.py --out data/dummy_chest_xray --per-class 6
python -m pytest tests/test_data.py
```

### Class balance (real dataset)

Output of `python data.py` against the real downloaded dataset:

```
train: {'NORMAL': 1341, 'PNEUMONIA': 3875} total=5216 pneumonia_to_normal_ratio=2.89
val:   {'NORMAL': 8, 'PNEUMONIA': 8} total=16 pneumonia_to_normal_ratio=1.00
test:  {'NORMAL': 234, 'PNEUMONIA': 390} total=624 pneumonia_to_normal_ratio=1.67
```

The official `val/` split only has 16 images, which is too small to validate
on reliably — `get_dataloaders()` pools `train/` and `val/` and re-splits
them (stratified by class, `resplit_val=True` by default) to fix this.
