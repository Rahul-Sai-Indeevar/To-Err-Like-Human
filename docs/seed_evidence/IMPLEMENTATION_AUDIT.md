# Implementation audit and stronger evidence

The existing implementation has a mathematically valid signal-processing connection. Whether it improves recognition reliably is a separate empirical question. This audit preserves the original paper-training defaults and proposes controlled opt-in experiments.

## Verified mathematical structure

The eight emotion probabilities form a signal on a label graph. Off-diagonal conductances are A_ij=1/W_ij and L=D-A is symmetric positive semidefinite. Its orthonormal eigenvectors define a graph Fourier transform. Parseval's identity, inverse reconstruction and the Dirichlet identity are tested.

The auxiliary fidelity quadratic with gamma=1 is (p-y)^T L+ (p-y), using the Laplacian pseudoinverse. Probability residuals have zero total mass, so exclusion of the DC eigenvalue is necessary. For a hard wrong prediction this energy equals effective resistance between the true and predicted label vertices. These are exact mathematical statements about the constructed graph; they do not measure physiological signals or validate psychological mechanisms.

The implemented loss is CE + alpha[(p-y)^T L+ (p-y) + beta p^T L p]. Inverse eigenvalues emphasize low graph-frequency residuals. This is not the ordinary positive-order Sobolev penalty and does not directly low-pass filter model outputs. The roughness term favors uniform probabilities in isolation. Adding it can bias probabilities; lower roughness alone is not better recognition. At gamma=0, the zero-mass residual quadratic becomes Brier error.

Under the unconstrained quadratic reference problem, the non-DC mode gain is 1/(1+beta*lambda^2). This reference excludes CE and softmax/nonnegativity constraints. At beta=.01 on the present graph, smoothing is mild. Absence of a large no-smoothing ablation effect would therefore be plausible.

Graph frequencies describe variation across label vertices. Spatial FFT frequencies describe intensity variation across image coordinates. Previous spatial-filter experiments already show that the small clean-image graph benefit does not imply uniform image-frequency robustness.

## Existing evidence and its limits

The full Kaggle experiment completed fifteen eight-epoch runs. The validation-selected alpha=.3 comparison uses only three independently trained seed pairs. ECC graph-minus-CE differences are +.1665, +.3801 and +.2568 percentage points, averaging +.2678. A paired Student t95 over those three differences is approximately [.0014,.5342], a fragile boundary result under normality assumptions. The exhaustive two-sided paired sign-flip p-value is .25, the smallest attainable value for three pairs. Accuracy's seed t95 includes zero. These statistics are descriptive because alpha was selected using the historical validation runs.

The previous positive image-bootstrap interval estimates uncertainty conditional on fitted checkpoints; it does not resolve variability across future training seeds. Repeating image draws cannot manufacture more independent model-training runs. The pilot subset, frozen-feature probe, full-data Kaggle run and spatial-filter study have distinct protocols and must remain separate.

No obvious implementation error was found in the graph objective or basic ECC/EMC calculations. Full-data paired CE/graph checkpoints selected the same epochs (5,7,6 respectively), so the observed difference is not simply a visibly unequal training budget. Training at eight epochs with fixed learning rate remains a limited optimizer budget; it is not proof that the outcome persists after convergence.

## Why the new controls are stronger

1. **Fresh paired seeds.** Twelve new seeds provide more actual training replications. Every method receives the same initial weights and explicit order/crop/flip plans within a seed. Validation chooses epochs; the coefficient remains fixed. Seed estimates remain conditional on the fixed data split.
2. **Matched isotropic quadratic.** Replace Q=L+ by q(I-J/K), q=tr(Q)/(K-1), and L by d(I-J/K), d=tr(L)/(K-1). This matches the average scales of both quadratics while removing emotional direction. Existing uniform-trace matching preserved tr(L), but not tr(L+); the old gamma=0 control also changed scale.
3. **Isospectral wrong-label graph.** Replace L by P L P^T while retaining true labels, output order and the correct scoring graph. Eigenvalues, both traces and all spectral moments stay unchanged. Alignment with the emotional labels changes. A preassigned permutation per seed avoids choosing a favorable wrong graph, but the paired difference then includes random-geometry variation.
4. **Separate real datasets.** FI and ArtPhoto, when available and with documented provenance/splits, test whether a pattern repeats beyond EmoSet. A resplit or capped EmoSet sample is not an independent dataset. Unequal dataset size and different learning difficulty mean results should not be pooled into a single effect estimate.

The isotropic and isospectral comparisons help distinguish generic quadratic regularization from emotion-aligned geometry. Failure to outperform either control leaves the mechanism unresolved. It does not prove that every graph method is ineffective.

## Statistical interpretation

The primary contrast is graph-minus-CE ECC separately within each dataset. Graph-minus-isotropic and graph-minus-permuted are secondary mechanistic contrasts. Holm adjustment across all dataset-by-three ECC comparisons controls the declared family under the sign-flip assumptions. Other metrics remain descriptive.

The paired Student interval treats seed pairs as independent experimental units. The exhaustive sign-flip test requires treatment-label exchangeability/sign symmetry under the null; it is not assumption-free. A crossed seed and class-stratified image bootstrap resamples seed pairs and shared image indices independently, preserving the same image draw across all models. With few seeds its percentile interval remains unstable.

A prespecified +.5 ECC-point reference separates a numerical difference from a chosen practically useful target. It is not an achieved-power claim. A nonsignificant test is not evidence of equivalence; no TOST equivalence study was prespecified. Prior benchmark inspection also makes these studies less clean than a new untouched benchmark.

## What verification can and cannot establish

The report checks probability normalization and finite values; exact labels/order against hashed CSVs; six metrics recomputed from saved predictions; complete histories; validation-only checkpoint selection; protocol and executed-runner hashes; common recorded initialization hashes; and independently reconstructed augmentation-plan hashes.

Agreeing recorded initialization hashes are evidence of pairing, not independent regeneration of the initial model. CSV checks do not independently rehash every image. GSEE is directly optimized by the graph objective, so lower GSEE is not independent validation. Neither a trained-model test nor algebra alone establishes human-like errors, a new best benchmark, or a causal account of visual emotion perception.
