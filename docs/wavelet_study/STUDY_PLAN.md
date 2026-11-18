# Wavelet multiresolution evidence study

Protocol authored 18 November 2026 before new wavelet outcomes. This is a local
locked exploratory protocol, not an external preregistration. The three existing
test splits have been inspected in earlier graph studies; new wavelet models
and their comparison settings are fixed before evaluating them.

## Question and primary comparison

Does allocating coarse and complementary fine-scale information to separate
CNN towers improve visual emotion recognition beyond equally sized towers
receiving RGB? Fusion minus RGB-dual ECC, separately on FI, EmoSet subset, and
ArtPhoto, is the primary comparison family (three Holm-adjusted exact paired
sign-flip tests). Improvement over a single RGB tower alone cannot identify a
wavelet contribution because the fusion model has twice the capacity/compute.

Secondary comparison family: fusion minus RGB, fusion minus low-dual,
low minus RGB, detail minus RGB (12 comparisons across three datasets). Report
ACC, ACC2, EMC, Brier, macro ECC, class patterns and training time descriptively;
avoid selecting a favorable metric after seeing test outcomes. A +0.5 ECC point
gain is a practical reference, not a formal equivalence or power boundary.

## Fixed models and signal processing

Use a two-level separable orthonormal Haar analysis/synthesis bank on already
ImageNet-normalized RGB z=(x-mu)/sigma. For a block [a,b;c,d], coefficients are
(a+b+c+d, a-b+c-d, a+b-c-d, a-b-c+d)/2. First band letter refers to filtering
along rows; second refers to columns. Repeat on LL for J=2. Reconstruct LL2
alone at the original 160x160 size as low=Pz; reconstruct all level-one and
level-two details as detail=(I-P)z. No clipping, absolute value, or per-image
rescaling. P is exactly a projection onto 4x4 block constants; LL2 coefficients
are four times block means. Low and detail are orthogonal and sum to z.

Six methods: rgb, low, detail (one ResNet18 tower each); wavelet_fusion
(low+detail), rgb_dual (RGB+RGB), low_dual (low+low), each two independent
ResNet18 towers with concatenated 512-dimensional features and one linear
eight-class head. Every dual model has exactly 22,361,224 trainable parameters;
single models have 11,180,616. All towers use the same ImageNet1K state per seed.
The single head is seeded per seed. Dual head blocks are W/2+delta and W/2-delta,
delta iid N(0,.01^2) from a local CPU generator seed900000+seed, with the original
bias. This preserves initial duplicate-RGB logits in real arithmetic and breaks
branch-gradient symmetry. Every dual method receives the same head. Towers
have independent parameter storage. No auxiliary polarity loss or graph loss:
all six train with identical eight-class cross entropy.

Fusion receives no new information relative to RGB. It tests a representation
and optimization hypothesis. Haar input decomposition does not remove CNN
downsampling aliasing or implement WTConv/WaveCNet inside the backbone. Critical
sampling is shift variant; low images are box projections, not ideal radial
Fourier low-pass images. Low-only tests do not establish brain mechanisms or
absence of useful detail information. Detail-only adaptation from ImageNet is
an important limitation when interpreting weak detail-only performance.

## Data, training, pairing

Reuse the previously deduplicated, provenance-audited fixed splits unchanged:
FI 4,777/800/1,593; EmoSet subset 4,800/800/1,600; ArtPhoto 468/155/160
train/validation/test. FI is a cleaned public mirror with ambiguous unavailable-
photo placeholders removed, and is not the canonical full FI benchmark.
ArtPhoto is a small exploratory transfer dataset. The EmoSet subset is capped
from available shards and is not a random sample of the full benchmark.

Eight seeds15-22, six methods, three datasets: 144 freshly trained models.
Twelve epochs each, batch64, AdamW lr1e-4, weight decay1e-4, gradient norm clip5.
Full backbone fine-tuning with CUDA mixed precision, deterministic algorithms,
no learning-rate sweep, early stopping or test-driven expansion. Resize192,
random crop160/flip.5 train; center crop160 validation/test; fixed ImageNet
affine. Identical seeded data order, crops and flips across six methods for
each seed. Transform is applied after crop/flip, with grid origin fixed at the
augmented image top-left. Select highest clean validation ECC, earliest tied
epoch. Record all epoch metrics and whether the selected epoch hits the budget
boundary. No test evaluation until all 144 training runs have completed.

One separate one-epoch FI fusion run with seed999 measures resource needs using
training and validation only. It is excluded from inference; changing batch or
budget afterward requires editing this plan and a fresh production output.

## Theory diagnostics and fixed sensitivity probes

Before outcome inspection verify perfect reconstruction, Parseval energy,
orthogonality, signed preservation, gradients, block-mean equivalence and
critical-sampling grid behavior. On a fixed training-only sample of up to32
images per class measure per-subband energy with both uncentered and spatially
centered summaries. For iid white noise the projection keeps an expected1/16
of total squared energy and detail15/16; verify numerically. Compare commutators
P(Tx)-T(Px): one-pixel circular horizontal shifts generally do not commute;
four-pixel shifts commute with P on the divisible periodic grid.

After selection, evaluate each frozen model on clean test inputs, circular
horizontal shifts1 and4 pixels, and unclipped additive Gaussian RGB noise
sigma=.03 (pixel range0..1), CPU generator612034 reset per model, identical
batch order. These are fixed descriptive sensitivity probes, not extra primary
hypothesis tests. Four-pixel commutation of preprocessing does not guarantee
CNN prediction invariance. For fusion also zero the detail or low input by
projecting the normalized tensor before the model; these are distribution-
changing sensitivity probes, not causal explanations or alternative selected
models. Do not tune settings from these outcomes.

## Verification and inference

Hash source code, input CSVs/caches, configuration, this plan, initial states,
augmentation plans and selected checkpoints. Archive executable source copies.
Independently compare every prepared source image to its training cache before
test evaluation. Reload every selected checkpoint and replay saved validation
probabilities bit-for-bit on the same device/runtime. Independently recompute
test/validation metrics and selection epochs. Save per-image predictions.

Training seeds, not images, are independent optimization replications on the
same fixed split. Report paired ECC differences, sample SD, Student t95,
leave-one-out sensitivity, exact sign-flip p (exchangeability/sign symmetry
assumption), and crossed seed plus shared class-stratified image bootstrap95
(1,500 resamples). Bootstrap image contributions are descriptive and do not
create more training replications. Do not pool datasets or call nonsignificance
equivalence. Existing split reuse, small seed counts, short training budget,
pretrained backbones, unresolved near duplicates and dataset-specific effects
limit generalization. Report negative findings without changing the hypothesis.

## Primary literature

- Mallat (1989), [A Theory for Multiresolution Signal Decomposition](https://www.di.ens.fr/~mallat/papiers/MallatTheory89.pdf): orthogonal multiresolution and filter banks.
- Fujieda et al. (2018), [Wavelet Convolutional Neural Networks](https://arxiv.org/abs/1805.08620): architectural precedent, not an implementation reproduced here.
- Li et al. (2020), [Wavelet Integrated CNNs for Noise-Robust Image Classification](https://arxiv.org/abs/2005.03337): wavelets inside CNN downsampling, a different intervention.
- Zhang (2019), [Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html): sampling and shift sensitivity.
- Finder et al. (2024), [Wavelet Convolutions for Large Receptive Fields](https://arxiv.org/abs/2407.05848): related wavelet architecture; this input-fusion study is not WTConv.
