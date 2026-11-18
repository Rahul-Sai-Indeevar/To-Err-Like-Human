# What the completed signal-processing experiments establish

The completed study provides a tested graph signal-processing formulation and stronger evidence about its limits. Across **144 trained models on three datasets**, it does **not establish a consistent recognition benefit or an advantage specifically attributable to emotion-aligned graph geometry** under the tested protocols. This is an uncertain, dataset-dependent empirical result, rather than proof that every small gain is training noise.

## Comparison and measured findings

Each dataset used twelve paired seeds (3–14), four objectives, eight epochs and full-backbone ResNet18 fine-tuning. The objectives were ordinary cross-entropy (CE), the proposed graph loss, a scale-matched isotropic quadratic, and an isospectral graph with incorrect emotion-label alignment. Initial weights, example order, crops and flips matched within each seed. Validation ECC selected checkpoints; the graph coefficient remained fixed at alpha=.3, gamma=1 and beta=.01.

The primary comparison asks whether graph training improves ECC over CE. The two controls ask whether emotional geometry contributes beyond generic quadratic regularization and spectral scale. Each comparison is made **within its dataset**. Equal epochs across datasets give different optimization budgets: 600 planned minibatches per model for the EmoSet subset and FI, versus 64 for ArtPhoto at batch64.

| Dataset | CE ECC | Graph ECC | Graph − CE points | Seed Student 95% interval | Crossed seed/image 95% interval | Positive seed pairs |
|---|---:|---:|---:|---|---|---:|
| EmoSet subset | 72.776 | 72.445 | −0.331 | [−0.6632, +0.0008] | [−0.9211, +0.2180] | 5/12 |
| FI | 65.376 | 65.925 | +0.549 | [−0.0127, +1.1111] | [−0.1093, +1.2638] | 8/12 |
| ArtPhoto | 50.445 | 50.273 | −0.172 | [−0.8213, +0.4775] | [−1.4697, +0.9623] | 7/12 |

ECC values are percentages and differences are percentage points. The Student intervals use paired training-seed differences. The crossed bootstrap independently resamples seed pairs and class-stratified test images, sharing each image draw across models. Test images are not counted as independent replications of training. Both approaches retain assumptions and remain conditional on these fixed splits.

**None of the nine declared ECC comparisons rejects at .05 after Holm correction.** FI's unadjusted exact sign-flip p=.0439 becomes .3955 after correction. Graph-minus-isotropic and graph-minus-permuted intervals also include zero on every dataset. The geometry-specific benefit therefore remains unestablished. Nonsignificance is not equivalence. The prespecified +.5 ECC-point practical reference is not an achieved-power claim, and the results do not establish an improvement exceeding it.

## Why this is a substantive signal-processing project

The eight class probabilities form a signal on a weighted **emotion-label graph**. With off-diagonal conductances A_ij=1/W_ij, the Laplacian L=D−A has an orthonormal eigenbasis U. Its graph Fourier transform separates variation across strongly connected labels into graph-frequency modes. These are neither physical hertz nor spatial frequencies in the input photograph.

For the zero-mass residual z=p−y, the fidelity penalty is

`z^T L+ z = sum_(k>0) lambda_k^(-1) |u_k^T z|^2`.

It emphasizes low graph-frequency residuals. The separate roughness penalty satisfies

`p^T L p = (1/2) sum_ij A_ij (p_i−p_j)^2`.

Parseval's identity, zero-DC handling and the pseudoinverse interpretation give explicit, testable SP content. For a hard incorrect class prediction, the residual energy equals the graph's effective resistance between the two labels. These mathematical identities concern the chosen label model; they do not establish human perceptual mechanisms.

The controls follow from this theory. Writing P0=I−J/8, the isotropic control replaces L+ by `[tr(L+)/7]P0` and L by `[tr(L)/7]P0`, independently matching both quadratic traces while removing emotional direction. A vertex-permuted graph preserves the entire eigenvalue spectrum while disrupting its alignment with the correct labels. All methods are evaluated using the same correct emotion graph. This is a direct test of the mechanism, beyond comparing an extra penalty against CE alone.

Mode-energy and probability-roughness diagnostics show that optimization changes the probability signal without guaranteeing better decisions. Mean roughness falls from .9969 to .9570 on the EmoSet subset and .4749 to .4172 on ArtPhoto, while their mean ECC decreases. FI roughness instead rises from .8118 to .8615. Modal changes are modest relative to seed variability. Brier and GSEE are somewhat lower for graph training, but GSEE is directly related to the optimized quadratic and is not independent validation. Uniform probabilities minimize roughness while losing class discrimination, so minimum energy alone cannot define a useful recognizer.

## Data quality, verification and scope

Actual train/validation/test counts are **4,800/800/1,600** for the capped EmoSet shard subset, **4,777/800/1,593** for FI, and **468/155/160** for ArtPhoto. Caps are maxima, not a promise of balance. ArtPhoto contains 783 usable images after exact deduplication from 806 raw entries. FI contains 21,808 usable images from 23,185 raw entries before caps.

Before any FI model outcomes, the preparation audit found **1,356 raw members of a shared placeholder group assigned conflicting labels**. All members were excluded, correcting the earlier first-label duplicate policy; original preparation records were preserved. Pinned source revisions, archive/shard hashes, preparation manifests and actual split counts are documented. Exact-hash checks do not eliminate re-encoded or near duplicates, and available cross-source hashes do not prove complete independence of photograph collections. These custom splits/caps are not official full-benchmark reproductions.

All 144 runs and 288 saved validation/test prediction arrays passed report verification: labels and order, finite normalized probabilities, recomputed decision/energy metrics, checkpoint selection, executed source snapshots, manifest/CSV hashes and matching paired initialization records. Augmentation-plan hashes were independently reconstructed. These checks verify the saved experiment artifacts; they do not independently repeat every model's training.

The separate **full-data replication with six fresh CE/graph seed pairs is complete** on 94,475 training, 5,904 validation and 17,715 test images. With alpha=.3 locked, graph-minus-CE ECC was **-0.136 points**, crossed seed/image 95% [-0.437, +0.131], with only 2/6 positive pairs. CE/graph mean ECC was 80.462/80.325. The historical +.268-point gain did not reliably repeat. This does not prove zero effect or isolate random seeds as the sole cause, because the old T4 environment/sampling implementation differ. See [the completed full-data conclusion](../full_seed_replication/CONCLUSION.md).

For presentation, the defensible contribution is: **derive an SP objective, derive controls that isolate its proposed mechanism, run matched experiments, and distinguish mathematical validity from demonstrated recognition improvement.** That connection remains meaningful even when the measured benefit is uncertain. Further model choices require validation and a genuinely new holdout; the inspected test sets should not become tuning targets.

Evidence: completed-study `summary.json`, `verification.json`, `dataset_provenance.json`, per-class/modal diagnostic tables, and the accompanying report figures.

An independent NumPy/Pillow audit additionally verified every one of the 15,153 prepared inputs against its exact training-cache row; see [input verification](input_verification.json). No GPU or training-runner import was used by that audit.

For the signal-processing explanation, see [the theory-to-experiment handout](THEORY_TO_EXPERIMENT.md), which maps each mathematical claim to its derived control, measured outcome and limitation.
