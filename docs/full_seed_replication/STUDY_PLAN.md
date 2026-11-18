# Full-data fresh-seed replication, 17 November 2026

The user explicitly selected this longer follow-up while the fixed 144-model
three-dataset study was still running. The completed medium EmoSet result was
known: graph-minus-CE ECC -0.331 points. This plan is recorded before any new
full-data model is trained or evaluated; it is not an external preregistration.

## Direct question and fixed design

Does the earlier full-data alpha=.3 improvement of +.268 ECC points repeat over
new training seeds, rather than only the original three fitted pairs?

- Recover the original Kaggle prepared photographs and CSVs from _output_.zip.
  Use all 94,475 train, 5,904 validation and 17,715 test images. Preserve the
  original assignments and source manifest; no new split or alpha search.
- Six fresh seed pairs, seeds 3 through 8. The original seeds 0/1/2 are separate
  historical context, not pooled into the primary fresh-seed analysis.
- CE versus CE+graph only. Alpha=.3, gamma=1, beta=.01 are locked from the
  original validation selection, irrespective of the new test outcomes.
- ImageNet ResNet18, train all parameters, AdamW lr1e-4, weight decay1e-4,
  gradient clip5, eight epochs, batch64, AMP forward and float32 loss.
- Each seed pair shares identical initial model state and explicit shuffled
  minibatch/crop/flip plans. Use the same transforms as the medium study:
  resized192 RGB, random crop160/flip for train, center160 for validation/test.
- Earliest checkpoint with maximum validation ECC; final test after selection.
  Complete all six pairs without significance-based stopping.
- One GPU job at a time, using the shared process lock. Input arrays remain
  memory-mapped on disk; transfer one batch to avoid exhausting host memory.
  Cache SHA256, source snapshots, epoch timings and AMP skipped-step diagnostics
  are saved. Resume only whole completed models under an unchanged protocol.
- A separate one-epoch validation-only timing run may precede training. It has
  no test evaluation and does not select model settings.

## Analysis and interpretation

The primary contrast is fresh graph-minus-CE ECC. Report all six differences,
mean, sample SD, Student t95, exact two-sided paired sign-flip p under its stated
symmetry/exchangeability assumption, and crossed seed/shared-stratified-image
bootstrap interval (1,500 draws). ACC/ACC2/EMC/Brier/GSEE are secondary.
The +.5 ECC-point practical reference remains unchanged. A nonsignificant
result is not proof of equivalence or zero effect. Six pairs give a minimum
two-sided exact p of .03125, so inferential resolution is limited.

Show the full-data single-contrast result explicitly and also a conservative
Holm adjustment across this contrast and the nine declared three-dataset ECC
comparisons. This prevents a favorable choice of comparison family after seeing
the full-data outcome. Do not pool datasets or old selected seeds into the
primary estimate. No post-test alpha choice is permitted.

The new pairs use a local RTX5050 and deterministic paired plans; the old runs
used T4 workers and a different random-sampling implementation/framework.
Thus this is an algorithmic fresh-seed replication on the identical prepared
dataset, not a bitwise repeat of the original trajectories. Test images have
already been inspected, so this directly studies repeatability conditional on
the established benchmark, not an untouched psychological or population study.
The two-method full-data comparison cannot by itself establish that emotional
geometry outperforms generic regularization; the four-method medium-data
controls address that separate mechanism question.
