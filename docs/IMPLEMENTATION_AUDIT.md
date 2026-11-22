# Implementation audit before Idea 1

Base: `21abe4fa8ea36fe23139e6c18a41d8491bcaffef` on the user's main branch.
Official comparison: `ZhaoChenxi-nku/ECC` at
`1c51f45cd1b17c12a009596109bb31c2f07bb84a`.

## What was present

- ResNet18/50/101 ImageNet transfer and supervised CSV datasets.
- ACC, polarity ACC2, ECC and EMC from a true-row/predicted-column confusion matrix.
- CE + alpha ListMLE, with stable class-index tie breaking.
- FixMatch-style weak/strong augmentation and EMC-adaptive threshold.
- An optional minimum-expected-cost decoder.
- Checkpoint saving, resume support, Kaggle helpers, and unit/smoke tests.
- Historical README FI runs: supervised CE 62.47 ACC / 73.42 ECC / 48.39 EMC;
  alpha=.2 ListMLE 62.09 / 74.39 / 55.92. The README itself reports that
  fixed-threshold SSL beat EMC-adaptive SSL in its compared n=800 run.

These historical scores are not new independently rerun results. No training
data or trained checkpoints were bundled in Git. The new experiment uses its
own downloaded real photographs and explicitly identified protocol.

## Verification

The previous wheel's full 8x8 W matches the literal evaluation matrix in the
official `train.py::eval_training` in wheel order. A pinned regression test now
checks all entries. EMC uses 1/(W-1) only off the diagonal; ECC uses 1/W.
The supplied threshold `e` is a dataset coefficient, not Euler's number.

This verifies the numeric matrix pattern. The author's eight-class label-index
mapping is loaded from an external dataset `info.json`, absent from that Git
repository; its end-to-end consistency cannot be independently checked here.
The new dataset importer checks its mirror's explicit class names against
numeric labels and computes W by name, avoiding dependence on that missing file.

The official code is partial. Its ListMLE training target matrix differs in
presentation from the evaluation matrix, and metric.py contains unrelated
distribution metrics. We do not claim exact author-training equivalence.
Existing total epochs, weight decay, split choice, and ListMLE aggregation
still carry the documented reproduction assumptions. S2-VER and the human
user study are absent.

The original resume smoke test checks that resumed training executes, not
bitwise equivalence of resumed and uninterrupted stochastic training. That
stronger claim is not established by the test's assertions.

## Idea 1 changes

- Dense, symmetric, zero-self-loop A_ij=1/W_ij and combinatorial Laplacian.
- GFT basis and nonnegative ordered graph frequencies; DC excluded safely.
- Inverse-frequency residual loss plus separate Dirichlet regularization.
- True heat-diffusion inference filter with validation-only scale selection.
- Graph error-energy/GSEE diagnostics, baseline and ablation comparisons.
- Tests for Parseval, Dirichlet, pseudoinverse, graph permutation invariance,
  heat-filter mass/DC preservation, gradients, and original-runner integration.

The graph objective uses actual distances and does not invent a strict ordering
between equally distant labels. This is a modeling distinction, not evidence
of an automatic accuracy or speed improvement. Original behavior is preserved
unless `loss.kind=graph` is set.

## Boundaries

Idea 2 is not implemented in this branch. Image-frequency claims in the supplied
text (such as low-pass images necessarily retaining polarity) remain untested.
Graph frequency is a property of variation over the emotion-label graph and
must not be confused with image FFT frequency or human neural measurements.
