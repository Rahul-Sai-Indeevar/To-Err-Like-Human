# Idea 1: graph Fourier analysis and spectral training

Work on branch `experiment/graph-spectral-emotion`. The main report is generated
at `outputs/graph_experiment/REPORT.md`. Data and model weights stay untracked.

## Setup (PowerShell, repository root)

Use a compatible Python/PyTorch/torchvision combination. CUDA is optional, but
recommended for full-backbone training. This machine used an RTX 5050 8 GB.

The local execution uses an isolated CPU dependency environment plus verified
CUDA 13.0 wheels unpacked at `work/cuda_runtime` (PyTorch 2.14.1+cu130 and
torchvision 0.29.1+cu130). `scripts/run_graph.ps1` adds that directory before
site-packages when present. Runtime versions in each protocol JSON are the
authoritative versions; a pip freeze of the base environment alone would list
the CPU packages. This is a local runtime arrangement, not a project dependency.
On another machine, install compatible CUDA packages normally instead.

For this existing checkout, use:

```powershell
./scripts/run_graph.ps1 -Stage test
./scripts/run_graph.ps1 -Stage features
./scripts/run_graph.ps1 -Stage probe
./scripts/run_graph.ps1 -Stage finetune
./scripts/run_graph.ps1 -Stage report
```

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]" matplotlib scipy pyarrow
$env:TORCH_HOME = "$PWD/data/torch_cache"
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
python scripts/download_emoset.py
python scripts/prepare_graph_data.py
python -m pytest -q
python -m ecc.graph_experiment --stage features
python -m ecc.graph_experiment --stage probe --epochs 60 --seeds 0 1 2 --device cpu
python -m ecc.graph_experiment --stage finetune --epochs 6 --seeds 0 1 2 --batch-size 32
python scripts/report_graph.py
```

After creating the environment, use `.venv/Scripts/python.exe` in place of
`python` unless that environment is activated. The download script pins its
resolved mirror revision in `provenance.json` and records SHA256 checksums.
To reproduce an existing data extraction exactly, preserve its source files,
provenance and CSV manifests. Mirror files may carry different licensing or
consent documentation from the official source; review source terms before
redistributing any photographs. Images are not committed or published here.

## Opt in through the original training interfaces

```powershell
python -m ecc.supervised --config configs/graph_emoset.yaml
```

The comparison runner uses AdamW for all matched methods. The config above uses
the original supervised runner's SGD and checkpointing. These are distinct
protocols: do not compare their scores as if all settings were identical.
`loss.kind=graph` also works with the existing semi-supervised runner, but this
extension's real-data experiments concern supervised classification only.
Existing configs without `loss.kind` preserve ListMLE behavior.

## Scientific design

- Primary model comparison: CE; CE + 0.2 ListMLE; CE + graph spectral residual
  and Dirichlet regularizer. Common initialization, seeds and validation ECC
  selection. Coefficients fixed before examining test results.
- Frozen-feature ablations: beta=0; gamma=0,beta=0 (Brier residual); uniform
  complete graph (label geometry removed). Frozen probes cannot establish that
  the backbone itself learns a better representation.
- Full fine-tuning changes all backbone parameters and separately tests the
  three main losses. Short six-epoch runs are a controlled medium-data pilot,
  not a full paper reproduction.
- Report graph eigenvalues, Fiedler/polarity correlation, GFT residual energy,
  GSEE, Brier error, Dirichlet energy, ACC, ECC, EMC and polarity accuracy.
- Keep GSEE separate from recognized benchmark metrics: it is an operational
  definition for this experiment, not an established psychometric measure.
- A graph on label similarities does not prove neural or psychological
  coarse-to-fine processing. GFT modes are not image Fourier frequencies.

## Interpretation corrections to the supplied proposal

The inverse-eigenvalue term emphasizes low-frequency residuals; it is not the
standard positive-order Sobolev norm. Dirichlet regularization acts on p, not
p-y, and alone favors uniform outputs. Eigenvector signs are arbitrary. Dense
A=1/W is not a ring-only graph. A Fiedler/polarity match is checked numerically,
not assumed for arbitrary polarity constants or graph constructions. ListMLE
has only eight positions here; a speed advantage must be measured, not asserted.
The report uses "graph error energy" rather than claiming stationary PSD.

## Files

- `src/ecc/graph.py`: graph, GFT basis, objective and opt-in dispatch.
- `src/ecc/metrics.py`: original scores plus explicitly defined graph metrics.
- `src/ecc/graph_experiment.py`: matched real-data training and predictions.
- `tests/test_graph.py`: transform identities, invariance and gradient tests.
- `tests/test_smoke_train.py`: original and graph training integration tests.
- `outputs/graph_experiment/`: reports, plots, environment, test log and runs.

Primary theory: Shuman et al. (2013), https://arxiv.org/abs/1211.0053.
Original paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2bb120e9a2cb9c2a50921b7f865c421-Abstract-Conference.html.
Dataset: https://github.com/JingyuanYY/EmoSet.
