# Fresh-seed evidence study, 17 November 2026

This plan is locked before inspecting outcomes of the new training runs. It is
a local protocol record, not a third-party preregistration. Previous EmoSet test
results have already been inspected and are treated as exploratory context.

## Questions

1. Does the validation-selected alpha=0.3 graph objective improve ECC repeatedly
   across fresh initialization and augmentation seeds on fixed data splits?
2. Does correct emotion geometry outperform a geometry-free quadratic of matched
   scale and an incorrectly labeled graph with exactly the same spectrum?
3. Do effects transfer to FI and ArtPhoto without selecting new hyperparameters?

## Fixed design

- Three datasets: existing balanced EmoSet pilot (4,800/800/1,600), new FI capped
  balanced sample (up to 600/100/200 images per class), and all usable ArtPhoto
  images with a fixed stratified 60/20/20 split. Report exact counts after dedup.
- FI and ArtPhoto come from the pinned public mirror revision
  476a96fdfc8330626bf60ce686008bed5605e5c0. These local splits are not official
  benchmark protocols. Downloaded images/checkpoints are excluded from git.
- Decoded-pixel duplicates are removed before splitting the new datasets;
  compare with known original-pixel hashes from the EmoSet pilot. Document
  remaining cross-dataset overlap limitations, especially near-duplicates.
- Twelve fresh paired training seeds 3 through 14. The fixed split is shared
  across all seeds; uncertainty does not include alternative training samples.
- Four methods: CE; CE+graph; CE+isospectral randomly mislabeled graph;
  CE+isotropic quadratic matched in both residual and smoothness trace.
- All auxiliary methods use alpha=.3, gamma=1, beta=.01. No alpha search.
  For the isospectral control, use the same preassigned random permutation for
  a given seed across datasets. Keep labels and evaluation W unchanged.
- ImageNet ResNet18, full fine-tuning, 8 epochs, AdamW learning rate 1e-4,
  weight decay 1e-4, gradient clip 5, batch size 64, AMP forward and float32
  loss. Resize 192, random crop 160 and flip for train, center crop for eval.
- All four methods in a seed share the exact initial model state, shuffled
  minibatches, crop coordinates and flips. Hashes verify this pairing.
- Select each checkpoint by highest validation ECC, earliest epoch on ties.
  Evaluate test only after that selection. No test-driven changes to methods.
- Single GPU, sequential training process with a shared process lock.
- Validation-only timing run uses a separate output directory and does not
  contribute to the study. No early stopping of the seed study for significance.

## Analysis

Primary metric: ECC in percentage points; ACC, polarity ACC2, EMC, Brier and
GSEE are secondary. Each independently trained seed pair is the replication
unit. Report per-seed differences, mean, sample SD, Student t95 interval,
exact two-sided paired sign-flip p under its exchangeability/symmetry assumption,
and leave-one-seed-out range. A crossed bootstrap resamples paired seeds and
shared class-stratified test images to expose image as well as seed uncertainty.
The three contrasts are graph-minus-CE, graph-minus-permuted, and graph-minus-
isotropic, separately for every dataset. Correct the nine ECC comparison
p-values by Holm; never pool seed runs from different datasets as independent
replications of a single population effect. An ECC gain of 0.5 points is a
predeclared practical reference, not a power guarantee or equivalence margin
demonstrated merely by a nonsignificant test.

There are 144 planned full-backbone runs, 48 per dataset. If an external source
fails or a run fails, preserve and disclose failures rather than silently
changing the protocol. A negative or uncertain result is a useful outcome.

## Interpretation limits

A repeated gain on a medium fixed subset is not proof of the same effect on the
full 94,475-image training set, a different backbone, or longer training.
An isospectral random-label graph control changes both label geometry and the
particular wrong geometry with seed. It tests the average preassigned random
control, not every possible graph. Finite test sets and data collection bias
remain. Graph frequency describes the label graph; spatial frequency describes
the photograph. Neither establishes a human perceptual mechanism.


The batch size was increased from the initial draft's 32 to 64 after separate
validation-only memory/timing checks, before any new study test evaluation.
The eight-epoch budget, methods, alpha, seeds and analysis remained fixed.

## Dataset quality amendment before FI training

The pinned FI mirror has 23,185 entries, not the commonly reported 23,308.
An audit found 1,356 copies of the same Flickr unavailable-photo placeholder,
spread across all eight labels. First-copy deduplication would retain one with
an arbitrary amusement label. Remove all conflicting-label exact-pixel groups,
including that placeholder; retain the original preparation manifests as an
audit trail. This is an input-quality correction before any FI training or test
outcome, not a model or hyperparameter change. Apply the same rule to ArtPhoto.

The 60/20/20 partition followed by class caps gives FI slightly fewer than the
nominal maximum: 4,777 train / 800 val / 1,593 test (the fear class has 577
train and 193 test). Keep that fixed partition rule and report actual counts;
do not force a new split to fill the cap. Describing FI as exactly balanced or
exactly 7,200 images would be inaccurate.
