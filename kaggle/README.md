# Kaggle Setup Guide for ecc-ver

This guide walks you through running the ECC/EMC paper re-implementation on **Kaggle GPU**.

## Overview

| Dataset | Source | Kaggle? |
|---------|--------|---------|
| **EmoSet-118K** | Public Kaggle: https://www.kaggle.com/datasets/georgeliil/emoset | Use directly |
| **FI** | Quanzeng You's page (OneDrive, 2016 AAAI) | Download locally, resize, upload as private dataset |

## Step 0: Prepare FI locally (one-time)

Download FI from the author's OneDrive (link in README.md / main paper). It arrives as class folders:
```
FI_raw/
  amusement/  anger/  awe/  contentment/  disgust/  excitement/  fear/  sadness/
```

Resize to 256×256 and create splits:

```bash
cd ecc-ver
pip install -e ".[dev]"
python scripts/resize_and_split.py \
    --root /path/to/FI_raw \
    --out /path/to/kaggle_fi \
    --val 0.05 --test 0.15 --seed 0
```

This produces:
```
kaggle_fi/
  FI/             # 256x256 class folders
  FI_splits/      # classes.txt, train.csv, val.csv, test.csv
```

Upload `kaggle_fi/` as a **private** Kaggle dataset (e.g. `your-username/fi-dataset-256`).

## Step 0b: Create a persistent checkpoint Dataset (recommended for EmoSet)

Kaggle free sessions are capped at ~12 hours and `/kaggle/working` is wiped when the container
recycles. To keep checkpoints alive across sessions, create a persistent Kaggle Dataset:

### Step 0b.1: Create the empty Dataset on Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) → **New Dataset** → **Private**
2. Name it something like `ecc-checkpoints`
3. **Do not upload any files** — leave it empty
4. Note your dataset slug: `your-username/ecc-checkpoints`

### Step 0b.2: Mount it in your notebook

1. Open your EmoSet notebook on Kaggle
2. In the sidebar, click **Add data** → select your private `ecc-checkpoints` dataset
3. Kaggle will mount it at `/kaggle/input/your-username/ecc-checkpoints/` (or similar)
4. In the notebook config cell, set:
   ```python
   PERSISTENT_DIR = "/kaggle/input/your-username/ecc-checkpoints"
   ```
   Replace `your-username/ecc-checkpoints` with your actual dataset path.

### Step 0b.3: How it works

- Every epoch, the training script copies `checkpoint.pt` and `best.pt` to `PERSISTENT_DIR`
- When the 12-hour session limit kills the container, `/kaggle/working` is wiped — but your
  checkpoint Dataset persists
- In the next session, mount the same Dataset and set `PERSISTENT_DIR` to the same path
- Training auto-detects `checkpoint.pt` in `PERSISTENT_DIR` and resumes from the last epoch

### Step 0b.4: Save a new Dataset version before session ends

**Important:** Before your session ends, save a new version of the checkpoint Dataset:

1. In the notebook, add a cell at the end:
   ```python
   # Save checkpoint Dataset version
   !kaggle datasets version -p /kaggle/input/your-username/ecc-checkpoints -m "Checkpoint after epoch X"
   ```
   Or do it via the Kaggle website: Dataset → **Save new version**
2. Next session, mount the **latest version** of that Dataset

This is **essential for EmoSet** (82K+ images, likely >12 hours per full run). FI should fit
within one session.

## Step 1: GitHub token for cloning (Kaggle Settings → Secrets)

Add a secret in Kaggle:
- **Name:** `GITHUB_TOKEN`
- **Value:** your GitHub personal access token (repo scope)

The notebook clones `https://github.com/Rahul-Sai-Indeevar/To-Err-Like-Human.git` using this
token so you always run the latest code.

## Step 2: Choose the right notebook

There are two separate notebooks — one per dataset — so results stay in separate folders:

| Notebook | Dataset | Results folder |
|----------|---------|----------------|
| `kaggle/train_fi.ipynb` | FI | `/kaggle/working/runs/fi_*` |
| `kaggle/train_emoset.ipynb` | EmoSet | `/kaggle/working/runs/emoset_*` |

1. New Notebook → set GPU to **P100 or T4** (Settings → Session options).
2. Add your datasets in the sidebar:
   - **FI notebook:** private FI dataset (resized 256×256).
   - **EmoSet notebook:** public EmoSet-118K + (optional) persistent checkpoint Dataset.
3. Copy the corresponding notebook into Kaggle.
4. **Edit the dataset slugs** at the top:
   ```python
   FI_DATASET_SLUG = "your-username/fi-dataset-256"
   EMOSET_DATASET_SLUG = "georgeliil/emoset"   # or whatever slug you found
   ```
5. For EmoSet, also set `persistent_dir` in the config cell to your checkpoint Dataset path.

## Step 3: Run

Run cells top to bottom (or use **Save and run all (commit)**):

1. Installs torch/torchvision + clones repo with GitHub token
2. Symlinks datasets into `data/`
3. Runs smoke tests (`pytest -q`)
4. Runs supervised training (alpha = 0 / 0.2 / 1.0)
5. Runs semi-supervised training (EMC-adaptive vs FixMatch baseline)
6. Creates `emoset_results.zip` for one-click download

**Tip:** Use **Save and run all (commit)** to execute everything without manually running each cell. The notebook will auto-resume from checkpoints if a previous session was interrupted.

## Step 4: Resume after session timeout

If the 12-hour cap kills the session:

- Re-run the same training cell — it auto-detects `checkpoint.pt` in the output dir **and** in
  `persistent_dir` (if configured) and resumes from the last epoch.
- Or force a specific checkpoint: `--set resume_from=/kaggle/working/runs/<run>/checkpoint.pt`

## Step 5: Download results

Before the session ends, download the results folder:
- **FI:** `/kaggle/working/runs/fi_*`
- **EmoSet:** `/kaggle/working/runs/emoset_*`

Each run contains: `best.pt`, `results.json`, `history.jsonl`, `config.json`, `checkpoint.pt`.

## Expected costs

| Run | Dataset | Est. time (P100) |
|-----|---------|------------------|
| Supervised ResNet50, 80 epochs | FI (23K images) | ~1–2 hrs per alpha |
| SSL, 50 epochs × 512 iters | FI (800 labels) | ~2–3 hrs per run |
| Supervised ResNet50, 80 epochs | EmoSet (82K+ images) | Likely >12 hrs — use persistent checkpoints + resume |

## File reference

| File | Purpose |
|------|---------|
| `kaggle/train_fi.ipynb` | FI Kaggle notebook |
| `kaggle/train_emoset.ipynb` | EmoSet Kaggle notebook |
| `scripts/resize_and_split.py` | Local FI resize + split generator |
| `src/ecc/kaggle_utils.py` | GitHub-token clone + persistent checkpoint helpers |
| `src/ecc/checkpoint.py` | Checkpoint save/load with resume |
| `src/ecc/supervised.py` | Supervised training (auto-resumes + persistent) |
| `src/ecc/ssl.py` | Semi-supervised training (auto-resumes + persistent) |
