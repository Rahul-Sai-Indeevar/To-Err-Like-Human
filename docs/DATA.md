# Prepared inputs

All split manifests have `path,label` columns and use exactly the order in their existing `classes.txt`. Check the archived fingerprints before comparing against the slide values. The model selects epochs on validation. Any use of a different manifest is a new experiment.

| Input | Default location |
|---|---|
| FI Haar train / validation / test | `data/wavelet/fi` |
| EmoSet subset Haar train / validation / test | `data/wavelet/emoset` |
| ArtPhoto Haar train / validation / test | `data/wavelet/artphoto` |
| Full EmoSet FFT test set | `data/emoset_full` |
| Three frozen CE weights | `checkpoints/ce_seed{0,1,2}/best.pt` |

Use a local copy of the YAML config to point each Haar dataset at an existing prepared directory. Pass the FFT data and checkpoint paths directly to its runner.

The historical FI reproduction used its own split and should be kept separate from the prepared FI Haar split. Capture source and split provenance when preparing a replacement from your own dataset. The FFT runner expects the already resized 192 × 192 source images from the archived protocol. The Haar scripts do their deterministic 192 × 192 resize and 160 × 160 crop from the images referenced by their manifests.

To reuse a working Python environment on the same computer, set `PYTHONPATH` to this repository's `src` directory and run the script with the existing environment's Python executable. In PowerShell, with `$originalCheckout` set to the old checkout directory:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& "$originalCheckout/.venv/Scripts/python.exe" scripts/run_wavelet_sp_checks.py --config local/wavelet.yaml --output runs/haar_checks
```

- [Full EmoSet test list and checkpoint hashes](../results/fft/input_fingerprints.json)
- [Haar split sizes, order and hashes](../results/wavelet/input_fingerprints.json)
