# ecc-ver

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
* **EmoSet** — https://github.com/JingyuanYY/EmoSet (check its access instructions/licence).

```bash
ecc-splits --root data/FI --out data/FI_splits            # 80/5/15 stratified random split
ecc-splits --root data/EmoSet/image --out data/EmoSet_splits   # or reproduce EmoSet's official split yourself
```

Class folder names are lower-cased and must be the 8 Mikel emotions. If you have official train/val/test
lists (EmoSet ships some), write them as `train.csv/val.csv/test.csv` with columns `path,label` plus `classes.txt`
(sorted class names) instead — `ecc-splits` is only a convenience.

## Run

```bash
# Step 1 - supervised, FI, ResNet50, pure CE then ListMLE (paper Table 4)
python -m ecc.supervised --config configs/fi_resnet50.yaml --set loss.alpha=0.0 output_dir=runs/fi_r50_a0
python -m ecc.supervised --config configs/fi_resnet50.yaml --set loss.alpha=0.2 output_dir=runs/fi_r50_a0.2
bash scripts/run_table4.sh                       # all backbones x alphas

# Step 2 - semi-supervised (paper 4.2.1): FixMatch baseline vs EMC threshold
python -m ecc.ssl --config configs/ssl_fi.yaml --set ssl.rule=fixed
python -m ecc.ssl --config configs/ssl_fi.yaml --set ssl.rule=ratio
bash scripts/run_ssl_sweep.sh

# Evaluate any checkpoint
python -m ecc.evaluate --config configs/fi_resnet50.yaml --ckpt runs/fi_r50_a0.2/best.pt
```

Each run writes `config.json`, `history.jsonl`, `best.pt`, `results.json` (test metrics with argmax and
with the MEC-decoding extension) under `output_dir`. Checkpoint selection uses the **val** split; the test
split is touched once at the end.

Paper targets (FI, ResNet50, Table 4): alpha=0 -> ACC 67.3 / ECC 77.0 / EMC 51.5;
alpha=0.2 -> 67.3 / 77.5 / 54.9; alpha=1.0 -> 65.7 / 76.2 / 57.3. Expect run-to-run noise of a few tenths.

## Things I could not verify (check these first against the paper PDF / official code)

| Item | Where | What I assumed |
|---|---|---|
| Circular order of Mikel's wheel | `emotion.py: WHEEL_ORDER` | Derived so that the paper's Appendix-B ranking for *excitement* is reproduced (tested). Other rows are unchecked. |
| EMC weight | `metrics.py: emc` | 1/(W-1) on misclassified cells (PDF extraction was ambiguous; this is what gives max 1). |
| Threshold rule, Eq. 5 | `ssl.py: adaptive_threshold` | PDF text is garbled. Implemented tau' = tau * eps / EMC, clipped to [0.7, 0.98]. |
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
