# From signal-processing theory to experiments

**Project question:** can the structure of emotion labels guide a classifier toward more useful errors, and does an improvement depend on that structure rather than an extra regularization term? We implemented the proposed graph objective, derived controls from its mathematics, and completed 144 matched training runs. The SP connection is explicit even though a reliable geometry-specific recognition benefit has not been established.

## The signal and the objective

For one photograph, the classifier produces an eight-entry probability vector p. These entries are a **signal on eight emotion vertices**. We construct off-diagonal conductances A_ij=1/W_ij from the paper's emotional-distance matrix W. With L=D−A and eigendecomposition L=UΛUᵀ, the graph Fourier coefficients are Uᵀp. A graph frequency describes variation across connected labels; it has no physical unit of hertz.

For one-hot target y, the implemented training objective is

`CE + alpha [ (p−y)ᵀ L+ (p−y) + beta pᵀ L p ]`,

with alpha=.3 and beta=.01 in the new study. L+ is the Laplacian pseudoinverse. The two quadratic terms have different spectral meanings and must be examined separately.

## Mathematical claim → derived experiment → evidence

| Mathematical claim | Control or test derived from it | Measured outcome | Interpretation and limit |
|---|---|---|---|
| **Orthonormal GFT and Parseval:** UᵀU=I; zᵀz equals the sum of squared coefficients of Uᵀz. | Test reconstruction/orthonormality. Recompute residual mode energies and check their sum against Brier error. Aggregate repeated-eigenvalue eigenspaces. | Algebraic tests pass; saved test-mode energies recover Brier for all 144 models. | This validates the transform and accounting of error energy. It does not establish better recognition. |
| **Zero DC and pseudoinverse:** z=p−y has zero total mass, so only non-DC modes contribute to zᵀL+z. For a hard error z=e_j−e_i, this is effective resistance between labels i,j. | Exclude the zero eigenvalue; test pseudoinverse and balanced-current identities. Check near-zero residual DC energy. | These numerical identities pass. No arbitrary eigenvector sign is assumed. | The electrical-network interpretation is exact for the constructed graph. It is not a measurement of brain activity or human emotion. |
| **Different spectral penalties:** zᵀL+z=sum_(k>0) lambda_k^(-1)(u_kᵀz)^2 emphasizes low-frequency residual error. pᵀLp=sum_k lambda_k(u_kᵀp)^2 penalizes high-frequency probability variation. | Inspect error energy by graph mode and report pᵀLp, Brier and GSEE alongside decision metrics. | Graph mean roughness decreases on the EmoSet subset and ArtPhoto, but increases on FI. Modal differences are modest relative to seed variability. | Gamma=1 is inverse-frequency weighting, not the ordinary positive-order Sobolev norm. The combined CE/softmax optimization need not reduce every diagnostic. GSEE is directly related to the trained objective. |
| **Geometry versus average quadratic scale:** let P0=I−J/8, q=tr(L+)/7 and d=tr(L)/7. The isotropic replacements Q_iso=qP0 and L_iso=dP0 match both traces. | Train CE+alpha[q times Brier + beta d times squared distance of p from uniform], using identical starting weights and data plans. | Graph-minus-isotropic ECC intervals include zero on all three datasets. | This removes emotion-aligned directions while matching both average quadratic scales. The current evidence does not isolate a geometry-specific advantage. It does not match every direction-dependent gradient. |
| **Isospectral label-mapping control:** PLPᵀ preserves all Laplacian eigenvalues while changing which labels occupy its vertices. | Preassign an incorrect, non-automorphism vertex permutation per seed. Keep output labels and evaluation graph correct. | Graph-minus-permuted ECC intervals include zero on every dataset; none rejects after the declared family correction. | Spectral scale alone cannot explain a difference between these objectives. Here alignment has no established benefit; the comparison also includes variation across the preassigned wrong graphs. |
| **Graph frequency and image frequency are distinct domains.** A two-dimensional image FFT measures intensity variation in cycles/pixel. | Independently low-pass/high-pass filter photographs and evaluate the frozen CE/graph models on identical inputs. | In the earlier full-test experiment, graph-minus-CE ECC was +.266 points on clean inputs but **−3.147** after low-pass filtering at .05 cycles/pixel. | Label-graph regularization does not guarantee spatial-frequency robustness. This intervention does not by itself identify the visual cues used or prove a theory of human perception. |
| **Probability energy and discrete decisions differ.** Energy can change while argmax stays unchanged. Uniform probabilities have pᵀLp=0. | Report ACC/ECC with probability diagnostics. Use uniform predictions as a mathematical counterexample to “smoother means better.” | A fixed argmax tie rule on uniform predictions gives **12.5% ACC** on the balanced eight-class subset, despite minimum roughness. | Class discrimination requires non-DC content. Low energy alone is not a valid success criterion; ECC depends on the selected class and its emotional distance from the reference. |

## What the new recognition experiments say

Each dataset used twelve paired seeds, four objectives and eight epochs of full-backbone fine-tuning. Validation selected checkpoints; alpha stayed fixed. The same initialization, example order, crops and flips were used within each seed. Comparisons are within a dataset: eight epochs gave 600 planned minibatches per model on the EmoSet subset and FI, but 64 on ArtPhoto.

| Dataset | CE ECC | Graph ECC | Graph − CE points | Seed Student 95% interval | Crossed seed/image 95% interval | Positive pairs |
|---|---:|---:|---:|---|---|---:|
| EmoSet subset | 72.776 | 72.445 | −.331 | [−.6632, +.0008] | [−.9211, +.2180] | 5/12 |
| FI | 65.376 | 65.925 | +.549 | [−.0127, +1.1111] | [−.1093, +1.2638] | 8/12 |
| ArtPhoto | 50.445 | 50.273 | −.172 | [−.8213, +.4775] | [−1.4697, +.9623] | 7/12 |

ECC values are percentages; differences are percentage points. None of the nine declared ECC contrasts rejects after Holm correction. The intervals do not establish equivalence or a gain exceeding the prespecified +.5-point practical reference. Test-image resampling cannot create independent training replications. These estimates remain conditional on the fixed splits and the stated statistical assumptions.

Generic regularization, changes in learned features, and initialization/augmentation/checkpoint variability are **possible explanations**, not established causes of the dataset-dependent results. The controls were designed to distinguish these possibilities; the observed uncertainty prevents a stronger mechanism claim. Further model choices should use validation and a new holdout rather than tuning against these inspected tests.

The separate **full-data replication with six fresh CE/graph seed pairs is complete** on 94,475 training, 5,904 validation and 17,715 test images. With alpha=.3 locked, graph-minus-CE ECC was **-0.136 points**, crossed seed/image 95% [-0.437, +0.131], with only 2/6 positive pairs. CE/graph mean ECC was 80.462/80.325. The historical +.268-point gain did not reliably repeat. This does not prove zero effect or isolate random seeds as the sole cause, because the old T4 environment/sampling implementation differ. See [the completed full-data conclusion](../full_seed_replication/CONCLUSION.md).

**Presentation takeaway:** the contribution is a chain of mathematical derivation, testable predictions, carefully matched controls and measured limits. The project demonstrates how SP theory can generate an experimental question without requiring the answer to be a performance improvement.

Evidence: [completed seed study and verification](REPORT.html), [graph-theory identities and original pilot](../graph_results/REPORT.html), and [full spatial-frequency experiment](../full_spatial_frequency/REPORT.html).
