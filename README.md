# ecc-ver

Latest wavelet evidence: **144 additional models**, six matched variants and eight
paired seeds on three prepared datasets. Fusion minus equally sized RGB+RGB ECC
was **-3.023 FI / -4.151 EmoSet / -0.968 ArtPhoto** points. FI and EmoSet lost in
all eight seed pairs; ArtPhoto was inconclusive. The projection reduced the fixed
white-noise sensitivity while giving lower clean and noisy absolute recognition.
See [wavelet findings](docs/wavelet_study/FINDINGS.md),
[full report](docs/wavelet_study/REPORT.html),
[noise and shift diagnostics](docs/wavelet_study/DIAGNOSTICS.html), and
[SP derivations and viva](docs/wavelet_study/THEORY_AND_VIVA.md).

Latest evidence: **156 new full-backbone runs** across three datasets and the separate full-data replication. The earlier +0.268 ECC-point gain did not reliably repeat: six fresh full-data pairs gave graph-minus-CE **-0.136 points**, crossed seed/image 95% [-0.437, +0.131]. See [the full-data conclusion](docs/full_seed_replication/CONCLUSION.md), [three-dataset findings](docs/seed_evidence/FINDINGS.md), and [the SP theory-to-experiment handout](docs/seed_evidence/THEORY_TO_EXPERIMENT.md).

## Graph signal processing extension

Idea 1 is implemented on branch `experiment/graph-spectral-emotion` with an
opt-in graph spectral loss, graph error-energy analysis, matched baselines and
real EmoSet experiments. See [protocol and reproduction](docs/GRAPH_EXPERIMENT.md).
Generated local findings are in `outputs/graph_experiment/REPORT.md`.

Independent re-implementation of **"To Err Like Human: Affective Bias-Inspired Measures for Visual
Emotion Recognition Evaluation"** (NeurIPS 2024) — ECC / EMC metrics, the ListMLE-based training
loss, and an EMC-adaptive threshold for FixMatch — plus a place to run your own improvements.

Official code (partial, used only for cross-checking): https://github.com/ZhaoChenxi-nku/ECC

## Layout

```
src/ecc/
  emotion.py     Mikel wheel, cost matrix W (Eq. 1), ListMLE rank targets
  metrics.py     ACC, ACC2, ECC (Eq. 3), EMC (Eq. 4)
  losses.py      L_c = CE + alpha * ListMLE
  decode.py      [extension] minimum-expected-cost decoding at test time
  data.py        CSV dataset, transforms, labeled-subset sampling
  splits.py      build train/val/test CSVs from class folders      (ecc-splits)
  supervised.py  supervised training, Tables 2-4                   (ecc-train)
  ssl.py         FixMatch + EMC-adaptive threshold, Table 1 (4.2.1) (ecc-ssl)
  evaluate.py    evaluate a checkpoint                              (ecc-eval)
configs/         fi_resnet50.yaml, emoset_resnet50.yaml, ssl_fi.yaml
scripts/         run_table4.sh, run_ssl_sweep.sh
tests/           metrics, losses, threshold, CPU end-to-end smoke tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # should pass without any dataset or GPU
```

## Data

You must download the datasets yourself (not included):

* **FI** (You et al., AAAI 2016) — folders per class: Amusement, Anger, Awe, Contentment, Disgust, Excitement, Fear, Sadness.
  Download from the author's OneDrive: https://onedrive.live.com/?id=%2Fpersonal%2Fab6522e29f6ed9a0%2FDocuments%2FCV%2D20200309T191449Z%2D001%2Fdatasets%2FEmotion%5Fdataset&listurl=%2Fpersonal%2Fab6522e29f6ed9a0%2FDocuments&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcURaYnBfaUltV3JocHBpZm50Z3hSdXdfNm8yV3c%5FZT11MlR2N0k&ga=1
  FI is not on Kaggle, so download it locally, resize to 256x256 with `scripts/resize_and_split.py`, then upload as a private Kaggle dataset.
* **EmoSet** — https://www.kaggle.com/datasets/georgeliil/emoset (118K version, used by the paper).
  Can be added directly on Kaggle as a public dataset.

```bash
# FI (local, then upload to Kaggle as a private dataset)
ecc-splits --root data/FI --out data/FI_splits            # 80/5/15 stratified random split

# EmoSet (on Kaggle, mount directly)
# python scripts/resize_and_split.py --root <emoset-root> --out <kaggle-out>  # optional resize
```

Class folder names are lower-cased and must be the 8 Mikel emotions. If you have official train/val/test
lists (EmoSet ships some), write them as `train.csv/val.csv/test.csv` with columns `path,label` plus `classes.txt`
(sorted class names) instead — `ecc-splits` is only a convenience.

## Running on Kaggle

The plan is to run training on **Kaggle GPU** (free P100/T4, ~30 GPU-hours/week, 12-hour session cap).

See `kaggle/README.md` for the full guide, or use `kaggle/train.ipynb` directly:

1. Add your private FI dataset (resized 256x256) and the public EmoSet-118K dataset to the notebook.
2. Edit the two dataset slugs at the top of `kaggle/train.ipynb`.
3. Run the cells top to bottom — supervised Table 4 runs, then SSL (EMC-adaptive vs FixMatch baseline).
4. Every epoch saves `checkpoint.pt`, so you can resume across the 12-hour session limit.
5. Download `/kaggle/working/runs/` before the session ends.


## Run

```bash
# Step 1 - supervised, FI, ResNet50, pure CE then ListMLE (paper Table 4)
python -m ecc.supervised --config configs/fi_resnet50.yaml --set loss.alpha=0.0 output_dir=runs/fi_r50_a0
python -m ecc.supervised --config configs/fi_resnet50.yaml --set loss.alpha=0.2 output_dir=runs/fi_r50_a0.2
bash scripts/run_table4.sh                       # all backbones x alphas

# Step 2 - semi-supervised (paper 4.2.1): FixMatch baseline vs EMC threshold
python -m ecc.ssl --config configs/ssl_fi.yaml --set ssl.rule=fixed output_dir=runs/ssl_fi_n800_fixed
python -m ecc.ssl --config configs/ssl_fi.yaml --set ssl.rule=ratio output_dir=runs/ssl_fi_n800_ratio
bash scripts/run_ssl_sweep.sh

# Evaluate any checkpoint
python -m ecc.evaluate --config configs/fi_resnet50.yaml --ckpt runs/fi_r50_a0.2/best.pt
```

Each run writes `config.json`, `history.jsonl`, `best.pt`, `results.json` (test metrics with argmax and
with the MEC-decoding extension) under `output_dir`. Checkpoint selection uses the **val** split; the test
split is touched once at the end.

Paper targets (FI, ResNet50, Table 4): alpha=0 -> ACC 67.3 / ECC 77.0 / EMC 51.5;
alpha=0.2 -> 67.3 / 77.5 / 54.9; alpha=1.0 -> 65.7 / 76.2 / 57.3. Expect run-to-run noise of a few tenths.

## Results (single run, FI, ResNet50)

### Supervised (Table 4)

| Run | ACC | ECC | EMC | ACC2 | Paper ACC / ECC / EMC |
|-----|-----|-----|-----|------|----------------------|
| alpha=0.0 (CE) | 62.47 | 73.42 | 48.39 | 84.70 | 67.3 / 77.0 / 51.5 |
| alpha=0.2 (ListMLE) | 62.09 | 74.39 | 55.92 | 87.69 | 67.3 / 77.5 / 54.9 |
| alpha=1.0 (ListMLE) | 60.05 | 72.90 | 54.95 | 87.52 | 65.7 / 76.2 / 57.3 |

Our results are within ~4–7 points of the paper on ACC and ~3 points on ECC. Single run, no hyperparameter tuning — the paper used 2× RTX 3090 with more careful tuning. Training curves: `docs/fi_training_curves.png`.

### Semi-supervised (Section 4.2.1)

| Run | Rule | ACC | ECC | EMC | ACC2 | best_epoch |
|-----|------|-----|-----|-----|------|------------|
| n=80 | EMC-adaptive | 29.08 | 48.94 | 45.24 | 69.00 | 46 |
| n=800 | EMC-adaptive | 34.11 | 53.28 | 48.43 | 71.01 | 42 |
| n=1600 | EMC-adaptive | 32.41 | 51.30 | 45.38 | 69.97 | 27 |
| n=800 | FixMatch fixed | 38.94 | 56.14 | 46.01 | 72.68 | 42 |

**Key finding:** FixMatch with a fixed threshold (baseline) outperforms the EMC-adaptive threshold on both ACC and ECC at n=800. The EMC-adaptive mechanism did not improve over the baseline in our reproduction — worth investigating further (e.g., different `eps`, EMA smoothing of the EMC signal).

## Things I could not verify (check these first against the paper PDF / official code)

| Item | Where | What I assumed |
|---|---|---|
| Circular order of Mikel's wheel | `emotion.py: WHEEL_ORDER` | All eight rows are regression-tested against the official train.py matrix at commit 1c51f45cd1b17c12a009596109bb31c2f07bb84a. |
| EMC weight | `metrics.py: emc` | 1/(W-1) on misclassified cells, divided by (N - Nc) (verified against paper PDF Eq. 4). |
| Threshold rule, Eq. 5 | `ssl.py: adaptive_threshold` | `tau' = tau * eps / EMC`, eps=0.5 FI / 0.4 EmoSet, clipped [0.7, 0.98] (verified against paper PDF Sec. 4.2.1). |
| CE -> combined loss switch | `supervised.py` | Section 4.2.2: train with pure CE then switch to Lc. New `loss.switch_epoch` config (0 = combined loss from start, matching Tables 2-4). |
| ListMLE aggregation | `losses.py` | Sum over list positions (standard ListMLE). `loss.listmle_reduction: mean` is available. |
| Weight decay, total epochs | configs | Not given in the paper (only: SGD, lr 1e-3 -> 2e-4 after epoch 60, bs 64, warm-up 1 epoch). |
| FI split | `splits.py` | FI has no official split; 80/5/15 random. |
| EMC for thresholding | `ssl.py` | Confusion matrix accumulated over the epoch's unlabeled batches (weak pseudo-label vs strong prediction), not a separate full pass. |

**Not implemented:** S²-VER integration (Table 1 lower block), the user study, and the t-SNE figure.
For S²-VER use https://github.com/exped1230/S2-VER and port `adaptive_threshold` and `CombinedLoss` into it.

## Ideas to improve on the paper

1. `decode.mec_decode` — minimum-expected-cost prediction (already wired into `results.json` as `test_mec_decode`).
2. Ramp `loss.alpha` during training instead of a constant.
3. Cost-aware soft labels (targets proportional to 1/W) as an alternative to ListMLE.
4. EMA-smooth the EMC signal used for thresholding; combine with per-class FlexMatch thresholds.
5. Sweep the polarity constant `loss.polarity_C` and the wheel definition.

Always report mean ± std over >= 3 seeds before claiming an improvement.

### Completed medium-data GSP experiment

The completed study used 7,200 balanced real EmoSet images, 21 frozen-feature
runs and 9 full-backbone runs. All 24 tests passed; all 30 saved runs were
verified and checkpoint replay matched saved predictions exactly.

Full fine-tuning mean over 3 seeds: CE ACC 58.90 / ECC 71.48 / EMC 51.30;
ListMLE 57.52 / 70.97 / 53.77; graph loss 60.02 / 72.32 / 51.80.
Graph-minus-CE ECC was +0.842 percentage points, with a class-stratified paired
image-bootstrap 95% interval [0.157, 1.501], conditional on these trained models
and the selected subset. Frozen-feature ECC differences were inconclusive.
These are six-epoch pilot results, not a full NeurIPS benchmark reproduction.

See [the complete theory and findings](docs/graph_results/REPORT.md),
[verification](docs/graph_results/verification.json), and
[reproduction instructions](docs/GRAPH_EXPERIMENT.md).

### Full EmoSet Kaggle recovery and spatial-frequency study

All 15 Kaggle runs completed eight epochs; the reported interruption was an
export-status bug (`st_size()` rather than the integer property `st_size`).
The repaired notebook is `kaggle/sp_full_alpha_sweep_fixed.ipynb`.

The full-data validation-selected graph coefficient is alpha=0.3. Its original
Kaggle test ACC/ECC are 71.252/80.491 versus CE 70.893/80.223 (three seeds).
A subsequent frozen-checkpoint study evaluated 16 image-frequency/phase
conditions on all 17,715 test images for six models: 96 evaluations, 1,700,640
image-model predictions. Graph regularization did not confer uniform spatial
robustness: at low-pass cutoff 0.05 cycles/pixel its ECC was 3.147 points below
CE. These are exploratory analyses on an already-inspected test split.

See [the complete recovery and findings](docs/full_spatial_frequency/REPORT.md),
[verification](docs/full_spatial_frequency/verification.json), and
`kaggle/sp_fft_followup.ipynb` for a Kaggle reproduction. The local report is
`outputs/full_spatial_frequency/REPORT.html`; run commands are included there.
All 29 tests pass. Original Downloads archives and notebooks are preserved.

### Baseline continuation safeguards

Checkpoint continuation now rejects incompatible experiment configurations before replacing outputs, restores the selected external run's best model, and records unverified legacy provenance. RNG and AMP scaler restoration remain absent, so exact uninterrupted continuation is not promised. See [the baseline audit and fixes](docs/seed_evidence/BASELINE_AUDIT.md).

### Fresh-seed evidence and reproduction

Run `scripts/run_seed_evidence.ps1` for the sequential three-dataset study and `scripts/run_seed_evidence.ps1 -Report` for its report. Prepare FI/ArtPhoto with `scripts/prepare_seed_datasets.py`; the report records pinned archives and custom split counts. `scripts/verify_seed_inputs.py --out outputs/seed_evidence/input_verification.json` independently checks every prepared image against the training cache. Use a fresh output directory when changing a protocol.

The full-data follow-up uses `scripts/run_full_seed_evidence.ps1 -Stage train` and `-Stage report`, six fresh seeds 3–8, and the complete original Kaggle split. Its cache is streamed from disk in batches. All test sets remain evaluation targets, with validation selecting checkpoints. Model weights and raw photographs remain local and excluded from Git.

The final code suite passes 78 tests. All twelve full-data checkpoints replay validation/test predictions bit-for-bit on the recorded local GPU/runtime. Use `scripts/run_full_seed_evidence.ps1 -Stage replay` before `-Stage report` for this audit. The local deliverable package is `outputs/sp_seed_evidence_bundle.zip`; it contains code, predictions, histories and input manifests, with raw photographs/checkpoints/caches excluded.

### Wavelet multiresolution experiment

The opt-in wavelet study separates normalized photographs into the two-level
Haar approximation and complementary signed details. Six models distinguish
representation from capacity: RGB, approximation, details, approximation+details,
RGB+RGB, and approximation+approximation. The three dual models have exactly
matched trainable parameter counts and paired asymmetric head initialization.

Run `scripts/run_wavelet_study.ps1 -Stage all` to train the fixed 144-model study
sequentially on one GPU, audit its input images, evaluate frozen models, reload
checkpoints, and produce `outputs/wavelet_study/REPORT.html`. The fixed settings
are in `configs/wavelet_study.yaml`; the theory and comparison plan are in
[the study plan](docs/wavelet_study/STUDY_PLAN.md). A separate validation-only
timing run is available with `-Stage benchmark`. Regenerate clean-result tables
with `-Stage report`. Changing recorded code, data, or settings requires a fresh
output folder. A killed partial model restarts from its initial state; finished
models are verified and skipped. No bitwise mid-training continuation is claimed.

The primary comparison is fusion against equally sized RGB+RGB, separately on
FI, the prepared EmoSet subset, and ArtPhoto. This is a locked exploratory study
on previously inspected splits, not a canonical full benchmark reproduction.
Haar reconstruction, Parseval energy, noise projection and shift sensitivity
provide direct signal-processing tests even if recognition does not improve.

The complete wavelet findings are in `docs/wavelet_study/FINDINGS.md`, including
negative matched-capacity effects and the noise/detail tradeoff. Run
`scripts/run_wavelet_study.ps1 -Stage diagnostics` to independently audit every
saved condition array and regenerate `DIAGNOSTICS.html`. The `all` stage also
runs this supplement and creates the synthetic theory figures. After recording
code and published documentation, build the reproducible evidence ZIP with
`python scripts/package_wavelet_study.py`; every included file is SHA-verified.
Raw dataset photographs, caches and model checkpoints stay outside the ZIP.
