# Paired Haar and RGB recognition protocol

The presentation compares two equal-capacity ResNet18 models on three fixed
datasets. One sends the two-level orthonormal Haar coarse and signed-detail
components to separate towers. The control sends the same RGB tensor to both
towers. Both train using eight-class cross-entropy with 22,361,224 parameters.

## Data and training

Use the prepared FI (4,777/800/1,593), EmoSet subset (4,800/800/1,600) and
ArtPhoto (468/155/160) training, validation and test splits. These are fixed
research splits. Use the archived split hashes to verify input identity.

Train both arms using seeds 15–22, 12 epochs, batch size 64, and AdamW with
learning rate and weight decay both 0.0001 and gradient clipping at norm 5.
Use deterministic kernels and CUDA mixed precision when running on GPU.
Resize to 192 × 192, train on seed-paired random 160 × 160 crops and flips,
and center-crop for validation and test. Apply ImageNet channel normalization
before Haar decomposition. Keep signed detail without clipping or rescaling.
Use the same pretrained ImageNet ResNet18 weights, dual classifier initialization,
and precomputed augmentation plan for both arms within a seed.

Select the earliest epoch attaining the highest clean validation ECC. Finish all
planned fits before scoring the test split. Reload the saved validation-best
state and replay the corresponding validation predictions before test inference.

## Inference and provenance

For each dataset, report the mean ECC for each arm and the paired difference
Haar minus RGB. Quantify uncertainty over seeds using a Student t interval and
an exact paired sign-flip test. The archived results also retain the original
Holm adjustment across the three datasets. The full historical study trained
six arms and 144 models. This runner retrains only the two arms in the slide.

The runner fingerprints the resolved configuration, source files, manifests,
decoded arrays, initialization, augmentation plans and checkpoints. The test
split is held fixed, so seed intervals describe variability across training
initializations on those splits.

[Theory and implementation](SIGNAL_PROCESSING.md) · [Archived seed results](../../results/wavelet/seed_results.csv)
