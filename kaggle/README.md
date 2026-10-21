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
recycles. To keep checkpoints alive across sessions:

1. Create a new **private empty** Kaggle Dataset (e.g. `your-username/ecc-checkpoints`).
2. Add it to your notebook as a dataset input.
3. In the notebook config, set `persistent_dir=/kaggle/input/your-checkpoint-dataset`.
4. Every epoch, `checkpoint.pt` and `best.pt` are copied there.
5. Before the session ends, save a new Dataset version from the notebook outputs (or via the
   Kaggle UI/API).
6. Next session, mount the latest version of that dataset — training resumes from the saved
   checkpoint.

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

Run cells top to bottom:
1. Installs torch/torchvision + clones repo with GitHub token
2. Symlinks datasets into `data/`
3. Runs smoke tests (`pytest -q`)
4. Runs supervised training (alpha = 0 / 0.2 / 1.0)
5. Runs semi-supervised training (EMC-adaptive vs FixMatch baseline)

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
