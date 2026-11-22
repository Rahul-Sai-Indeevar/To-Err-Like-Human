# To Err Like Human · Haar Wavelet Study

This project studies how image frequencies affect visual emotion recognition. The baseline predicts one of eight emotions that a photograph can evoke. Exact-label accuracy counts correct predictions. **Emotion Confusion Confidence (ECC)** also gives partial credit when the prediction is emotionally close to the reference label. **Emotional Misclassification Confidence (EMC)** measures closeness among the mistakes.

The signal-processing experiments start with a spatial FFT low-pass sweep, then test how a two-level Haar filter bank separates coarse structure from signed detail. The final comparison feeds those components to two independent branches and compares them against an equal-capacity model whose two branches both receive RGB.

## Repository structure

| Path | What lives here |
|---|---|
| [`src/ecc/`](src/ecc/) | Reusable Python package. [Emotion costs and ordering](src/ecc/emotion.py), [metrics](src/ecc/metrics.py), and [ranking loss](src/ecc/losses.py) define how the project trains and scores predictions. [Supervised](src/ecc/supervised.py), [semi-supervised](src/ecc/ssl.py), and [evaluation](src/ecc/evaluate.py) modules implement the baseline entry points. |
| [`src/ecc/spatial_frequency.py`](src/ecc/spatial_frequency.py) | Two-dimensional FFT filtering on the pixel grid. The low-pass path multiplies each channel’s spectrum by a radial Butterworth response and transforms it back to pixels. |
| [`src/ecc/wavelets.py`](src/ecc/wavelets.py) | Orthonormal Haar analysis and synthesis, multilevel decomposition, coarse and signed-detail reconstruction, and the matched recognition models. |
| [`configs/`](configs/) | YAML configurations for data paths, optimization, model settings, and the locked wavelet study. Use [`wavelet_study.yaml`](configs/wavelet_study.yaml) to see the dataset split paths, methods, seeds, and training budget. |
| [`scripts/`](scripts/) | Experiment entry points. [Haar diagnostics](scripts/run_wavelet_sp_checks.py) verify tone response, alias cancellation, and grid averaging. [Wavelet training](scripts/run_wavelet_study.py) fits the paired classifiers, and the [spatial-frequency runner](scripts/run_spatial_frequency.py) evaluates archived checkpoints on filtered inputs. |
| [`kaggle/`](kaggle/) | Notebooks and utilities for the archived GPU runs, including the full EmoSet sweep used to obtain the frozen checkpoints. |
| [`tests/`](tests/) | Checks for metric and loss behavior, transform identities, and training and checkpoint workflows. |
| [`docs/wavelet_study/`](docs/wavelet_study/) | The [training protocol](docs/wavelet_study/STUDY_PLAN.md), [signal-processing derivations](docs/wavelet_study/SIGNAL_PROCESSING.md), [filter measurements](docs/wavelet_study/wavelet_sp_experiments.md), and [recognition findings](docs/wavelet_study/FINDINGS.md). |
| [`docs/full_spatial_frequency/`](docs/full_spatial_frequency/) and [`docs/seed_evidence/`](docs/seed_evidence/) | Saved results from the spatial FFT sweep and earlier experiment runs. |
| [`pyproject.toml`](pyproject.toml) | Package dependencies, command-line entry points, and test discovery settings. |
| `data/`, `runs/`, `outputs/`, `work/` | Local datasets, checkpoints, generated outputs, and temporary files. These directories are Git-ignored. |

## Results and plots

### 1. FI baseline reproduction

| Historical FI run | Accuracy | ECC | EMC |
|---|---:|---:|---:|
| Supervised cross-entropy | 62.47% | 73.42% | 48.39% |
| Cross-entropy + ListMLE (α = 0.2) | 62.09% | 74.39% | 55.92% |
| Fixed threshold, 800 labels | 38.94% | 56.14% | 46.01% |
| EMC-adaptive threshold, 800 labels | 34.11% | 53.28% | 48.43% |

These are descriptive single runs. The ranking loss produced closer mistakes on average. In the 800-label run, the fixed threshold scored higher on accuracy and ECC.

### 2. Spatial FFT low-pass filtering

The same three frozen cross-entropy models evaluated **17,715 full EmoSet test photos**. Mean ECC was about **80.2%** on clean inputs, **45.74%** at a cutoff of **0.05 cycles/pixel**, and **71.87%** at **0.20 cycles/pixel**. Lower cutoffs suppress more fine detail. [Full sweep and conditions](docs/full_spatial_frequency/REPORT.md) · [Results table](docs/full_spatial_frequency/summary.csv)

![Spatial frequency response on frozen EmoSet models](docs/full_spatial_frequency/frequency_response.png)

### 3. Haar filter-bank diagnostics

The Haar transform splits squared signal energy among its bands. Across 99 controlled tone cases the measured energy fractions matched theory to floating-point precision. The two reconstructed paths contributed opposite signs at the folded FFT bin and canceled when added. Averaging the 16 possible 4 × 4 grid alignments reduced measured shift error to **0%** for the tested one-pixel circular shift across **534** FI, EmoSet, and ArtPhoto crops. These measurements describe filtering, and use the [saved diagnostics](docs/wavelet_study/wavelet_sp_results.json).

![Measured Haar frequency response, alias cancellation, grid averaging, and noise retention](docs/wavelet_study/wavelet_sp_checks.png)

### 4. Matched-capacity Haar recognition

The primary comparison pairs eight training seeds per dataset and gives both models the same parameter count. The table uses mean held-out ECC.

| Dataset | RGB + RGB | Coarse + detail | Haar − RGB | Paired 95% interval |
|---|---:|---:|---:|---:|
| FI | 66.33% | 63.31% | −3.02 points | [−4.12, −1.92] |
| EmoSet subset | 73.10% | 68.95% | −4.15 points | [−4.85, −3.45] |
| ArtPhoto | 51.62% | 50.66% | −0.97 points | [−2.71, +0.78] |

The fixed Haar split scored lower on FI and the EmoSet subset. ArtPhoto’s paired interval includes zero. The recognition experiment used the original fixed grid. The 16-grid average was evaluated in the filter diagnostics above. [Per-seed results](docs/wavelet_study/seed_results.csv) · [Paired effects](docs/wavelet_study/summary.csv)

![Paired ECC differences for the Haar fusion and matched RGB models](docs/wavelet_study/primary_effects.png)

## Running the project

From the project root, install the package and run the CPU tests:

```console
python -m pip install -e ".[dev]"
python -m pytest -q
```

For experiments that use images, prepare `classes.txt` and the `train.csv`, `val.csv`, and `test.csv` manifests. Each CSV uses `path,label` columns with image paths relative to its dataset root. Use the prepared splits specified in the configuration when reproducing the tables. The wavelet study uses CUDA for the full training runs.

```console
python scripts/run_wavelet_sp_checks.py --repo . --output outputs/wavelet_sp
python scripts/run_wavelet_study.py --config configs/wavelet_study.yaml --out outputs/wavelet_study --stage all --device cuda
```
