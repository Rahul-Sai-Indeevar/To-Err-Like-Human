# Final full-data replication finding

The earlier +0.268 ECC-point gain was not reliably repeated in the fresh-seed study. Across six new paired seeds on the original full prepared dataset, graph loss averaged **-0.136 ECC points versus CE**, with a crossed seed/image 95% interval **[-0.437, +0.131]**. Only 2/6 pairs improved; exact paired sign-flip p=0.21875. The seed Student interval was [-0.386, +0.114].

| Model | Mean ACC | Mean ECC |
|---|---:|---:|
| CE | 71.325 | 80.462 |
| Graph alpha=.3 | 71.118 | 80.325 |

The observed ECC seed differences were +0.0033, -0.1620, -0.0005, -0.6027, -0.0746, +0.0186. The leave-one-pair-out mean remains negative throughout [-0.1673, -0.0430], so the mean reversal is not removed simply by excluding the worst pair. These are descriptive sensitivity results, not proof of a negative population effect.

The result is consistent with small training-run-dependent effects and does not establish a repeatable positive benefit. It does **not prove an exactly zero effect** or that random seeds alone caused the historical difference: the original T4 environment and random-sampling implementation differ from the local deterministic paired protocol. The original three selected pairs remain historical context and were not pooled into the fresh six.

The crossed interval's upper bound is below the prespecified +0.5-point practical reference, under the stated sampling assumptions. This is not a formal equivalence/TOST result. The global ten-comparison Holm sensitivity also does not reject the full-data null. No post-test coefficient search was performed.

## Signal-processing conclusion

The GFT, Parseval, inverse-Laplacian/effective-resistance and Dirichlet formulations remain mathematically valid. A valid SP objective does not guarantee better held-out recognition. The twelve-seed medium-data controls did not establish an advantage of correct emotional geometry over matched isotropic or isospectral permuted-label penalties. The full-data comparison tested repeatability against CE; it did not add those mechanism controls. The earlier spatial-frequency study also showed that label-graph regularization does not guarantee image-frequency robustness.

A defensible project claim is therefore: derive an SP formulation and controls from theory, measure its probability/decision effects, and demonstrate the limits of an initially promising small gain. A new method should be selected using validation and evaluated on a fresh holdout, rather than tuned against these inspected tests.

## Verification and reproduction

All twelve full-data models completed eight epochs on 94,475 training images; validation/test contain 5,904/17,715 images. Report verification checked 24 saved prediction arrays, paired plans, source/data hashes, selected epochs, metrics, input-cache hashes and AMP diagnostics. An independent CPU audit matched all 118,094 source inputs to cache rows.

All twelve saved checkpoints were reloaded on the same GPU and reproduced both validation and test probabilities **exactly**: maximum absolute probability difference 0.0, argmax agreement 100%. The final project suite passed **78 tests**. Raw images, cache arrays and checkpoints remain local and excluded from Git.

See [the complete report](REPORT.html), [checkpoint replay](checkpoint_replay.json), [input audit](input_verification.json), [verification](verification.json), [locked plan](STUDY_PLAN.md), and [the SP theory handout](../seed_evidence/THEORY_TO_EXPERIMENT.md).
