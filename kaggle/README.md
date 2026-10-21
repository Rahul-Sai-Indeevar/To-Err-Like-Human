# Kaggle Setup Guide for ecc-ver

This guide walks you through running the ECC/EMC paper re-implementation on Kaggle.

## Overview

| Dataset | Source | Kaggle? |
|---------|--------|---------|
| **EmoSet-118K** | Public Kaggle (you found it) | Use directly |
| **FI** | Quanzeng You's page (OneDrive, 2016 AAAI) | Download locally, resize, upload as private dataset |

## Step 0: Prepare FI locally (one-time)

Download FI from Quanzeng You's page (link in the 2016 AAAI paper). It arrives as class folders:
```
FI_raw/
  amusement/
  anger/
  awe/
  contentment/
  disgust/
  excitement/
  fear/
  sadness/
```

Resize to 256×256 and create splits:

```bash
cd ecc-ver
pip install -e ".[dev]"      # installs Pillow, torch, etc.
python scripts/resize_and_split.py \
    --root /path/to/FI_raw \
    --out /path/to/kaggle_fi \
    --val 0.05 --test 0.15 --seed 0
```

This produces:
```
kaggle_fi/
  FI/
    amusement/  anger/  ...   (256×256 JPEGs)
  FI_splits/
    classes.txt
    train.csv   val.csv   test.csv
```

Upload `kaggle_fi/` as a **private** Kaggle dataset (name it something like `fi-dataset-256`).

## Step 1: Create a Kaggle notebook

1. New Notebook → set GPU to **P100 or T4** (Settings → Session options).
2. Add your private FI dataset and the public EmoSet-118K dataset in the sidebar (the "Add data" button).
3. Copy `kaggle/train.ipynb` into the notebook.
4. **Edit the two dataset slugs** at the top of the notebook:
   ```python
   FI_DATASET_SLUG = "your-username/fi-dataset-256"
   EMOSET_DATASET_SLUG = "username/emoset-118k"
   ```

## Step 2: Run

Run the cells top to bottom:
1. Installs torch/torchvision
2. Installs `ecc-ver` (the repo code must be available — either clone from GitHub or upload as a Kaggle dataset)
3. Symlinks datasets into `data/`
4. Runs smoke tests (`pytest -q`)
5. Runs supervised training (alpha = 0 / 0.2 / 1.0)
6. Runs semi-supervised training (EMC-adaptive vs fixed threshold)

## Step 3: Resume after session timeout

Kaggle sessions are capped at ~12 hours. Every epoch saves `checkpoint.pt` in the run's `output_dir`. If a session dies:

- Re-run the same training cell — it auto-detects `checkpoint.pt` and resumes from the last epoch.
- Or force a specific checkpoint: `--set resume_from=/kaggle/working/runs/<run>/checkpoint.pt`

## Step 4: Download results

Before the session ends, download `/kaggle/working/runs/`:
- `best.pt` — best-on-val model weights
- `results.json` — test metrics (ACC, ECC, EMC, ACC2)
- `history.jsonl` — per-epoch training log
- `config.json` — exact config used

## Expected costs

- **FI supervised** (ResNet50, 80 epochs, 23K images): ~1–2 hours on P100 per alpha
- **FI SSL** (800 labels, 50 epochs × 512 iters): ~2–3 hours per run on P100
- **EmoSet** (82K+ images): much heavier — a full 80-epoch run will likely exceed one Kaggle session; use resume or reduce `optim.epochs`

## File reference

| File | Purpose |
|------|---------|
| `kaggle/train.ipynb` | Kaggle notebook (edit slugs, run cells) |
| `scripts/resize_and_split.py` | Local FI resize + split generator |
| `src/ecc/checkpoint.py` | Checkpoint save/load with resume |
| `src/ecc/supervised.py` | Supervised training (auto-resumes from `checkpoint.pt`) |
| `src/ecc/ssl.py` | Semi-supervised training (auto-resumes from `checkpoint.pt`) |
