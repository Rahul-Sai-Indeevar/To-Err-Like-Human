# Implementation audit: baseline and paper-reproduction scope

Read-only audit of the local repository on 17 November 2026. The findings below
concern the legacy supervised/SSL entry points, not the separate fresh-seed GPU
runner. No training code or running experiment was modified during this audit.
Line references describe the files as inspected, before any subsequent fixes.

## What is present and sound

- `emotion.py` builds the eight-label wheel distance matrix and polarity vector.
  `tests/test_graph.py:8` checks all distance rows against the pinned official
  evaluation matrix. The older README assumption that only one row was checked
  (`README.md:128`) is stale.
- `metrics.py` computes ACC, polarity ACC2, ECC and EMC from confusion matrices
  with rows=true and columns=predicted. EMC conditions on errors; its explicit
  perfect-classifier convention is 100%, rather than division by zero.
- `losses.py:9` implements a numerically stable Plackett-Luce/ListMLE likelihood
  through reverse log-cumulative-sum-exp. `CombinedLoss` adds it to CE. Sum versus
  mean over positions changes its effective coefficient by eight for eight labels.
- `supervised.py` has pretrained ResNet training, learning-rate warm-up/decay,
  validation checkpoint selection, and final argmax/MEC evaluation.
- `ssl.py` has labeled/unlabeled sampling, weak pseudo-labels, confidence masking,
  strong-view CE, an EMA model and EMC-adaptive thresholding. True unlabeled
  labels are used only in the monitoring statistic (`ssl.py:162`), not the loss.
- `decode.py:12` correctly minimizes expected cost `1-1/W`; equivalently it
  maximizes expected ECC utility for the supplied probabilities. It is explicitly
  an extension, not the original paper method or independent psychological evidence.
- The new graph loss, frozen-feature/finetuning pilots and FFT intervention study
  are optional extensions. The fresh-seed runner is separate from the issues below.

## Material legacy defects

1. **Automatic persistent-directory resume is unreachable when no local checkpoint
   exists.** `supervised.py:72` and `ssl.py:126` convert a missing local checkpoint
   to the truthy string `"None"` before using `or`. This prevents trying the
   persistent checkpoint. An explicit `--resume` still takes precedence.

2. **Reusing an output directory can silently relabel an old completed run as a
   different experiment.** Both entry points automatically load an existing local
   checkpoint without comparing the saved configuration (`supervised.py:74`,
   `ssl.py:128`), after overwriting `config.json` (`supervised.py:39`, `ssl.py:81`).
   The README's two commands at lines 86-87 change only `ssl.rule`, retaining the
   same default output directory from `configs/ssl_fi.yaml:3`. If the fixed run
   completed 50 epochs, the ratio command can skip its loop entirely and save
   fixed-model scores labeled `ratio`. The sweep shell script does assign separate
   directories and avoids this particular trigger. Historical README results
   cannot be declared affected without their original artifacts.

3. **Explicit CE selection can be overridden into ListMLE.** `graph.py:107`
   deliberately constructs `CombinedLoss(alpha=0)` for `loss.kind=ce`, but
   `supervised.py:84` unconditionally sets `crit.alpha` from `loss.alpha` each
   epoch. A config with `kind=ce` and nonzero alpha therefore trains ListMLE.
   This does not affect the fresh-seed runner's direct `CrossEntropyLoss`.

4. **External resume does not recover the best model into the new output folder.**
   Resume restores the historical best score/epoch but not the corresponding
   `best.pt`. Final evaluation unconditionally loads local `out/best.pt`
   (`supervised.py:112`, `ssl.py:187`). If no new epoch beats the old best, or
   training is already complete, resuming into a fresh output folder can fail or
   use an unrelated pre-existing best file. Saving the last model in checkpoint
   is not sufficient to reconstruct an earlier best model.

5. **Resume is continuation, not exact trajectory reproduction.**
   `checkpoint.py:25` saves model/optimizer/epoch and optional scheduler/extra
   state, but no Python/NumPy/Torch/CUDA RNG state. The supervised AMP scaler is
   not saved or restored (`supervised.py:63`, `supervised.py:106`). Restarting
   re-seeds before model construction and cannot recover the interrupted random
   data-order/augmentation stream. The resume smoke test at
   `tests/test_smoke_train.py:64` only checks that execution succeeds and the score
   is in range; despite its docstring, it never compares uninterrupted weights
   or predictions. `checkpoint.py:5` therefore overstates exact recovery.

## Scientific reproduction gaps and interpretation limits

- Historical FI/SSL README tables are explicitly single-run results. They are not
  a fresh independent reproduction in this study. Hardware differences alone do
  not explain their gap to the paper (`README.md:111`). The original split,
  optimization details and run artifacts need checking before such an explanation.
- `configs/fi_resnet50.yaml:21` and line 23 explicitly mark weight decay and total
  epochs unverified. The SSL config uses reduced batch size, unlabeled ratio and
  iterations, with an uncertain training budget (`configs/ssl_fi.yaml:16-21`).
  These are valid implementation choices but not exact-paper training protocols.
- `build_rank_matrix` imposes stable class-index tie breaking on equal emotional
  distances. Full ListMLE treats those artificial within-tie orderings as strict
  preferences. Sum/mean reduction, ties and the extra CE component are separate
  reproduction questions, even though standard ListMLE itself is correctly coded.
- The adaptive threshold is driven by weak-versus-strong model disagreement over
  all unlabeled batches (`ssl.py:151`), including masked-out examples. Its EMC is
  an internal consistency proxy, not measured emotion-label accuracy. The next
  threshold uses the configured base `s['tau']` (`ssl.py:166`), rather than the
  preceding adaptive value: this distinction should be described precisely.
- `splits.py` makes stratified file splits but has no image-content deduplication
  or near-duplicate grouping. `data.py:47` assumes exact class names and a plain
  CSV header, and `data.py:61` does not normalize EXIF orientation. The dedicated
  new experiment preparation is more defensive. The newly discovered FI
  conflicting-label duplicate group makes data auditing practically relevant.
- S2-VER integration, the human user study and t-SNE reproduction remain absent,
  as already stated by `README.md:137` and `ssl.py:9`.

## Consequence for the fresh-seed study

The standalone runner locks source/runtime/data/settings, uses separate output
directories, matches initialization and augmentation plans within seed pairs,
selects by validation ECC, and evaluates the correct unpermuted label graph. It
does not call the legacy supervised/SSL checkpoint-resume path. The defects above
do not invalidate its saved outcomes.

Its own limits remain: fixed dataset splits, a capped eight-epoch budget, partial
EmoSet/FI training sets, only twelve initialization/augmentation seeds, and small
ArtPhoto validation/test sets. All backbone parameters are trained; the protocol
phrase "fixed backbone" refers only to the common pretrained initialization.
Within-dataset paired comparisons are valid; different dataset sizes imply
different numbers of optimizer updates. A non-significant result means a benefit
was not established, not proof that the true effect is exactly zero.

## Resolution on 17 November 2026

The persistent-path selection, configuration collision, best-checkpoint recovery,
and explicit-CE dispatch defects described above are now fixed and regression
covered. README SSL examples use separate output directories. These defects
were in the baseline CLI runners, not the isolated graph experiment or the
fresh-seed runner. No saved historical numerical result has been relabeled or
recomputed by these fixes. RNG/AMP-scaler restoration remains absent, so exact
uninterrupted resume is explicitly not promised. See checkpoint.py and
resume_provenance.json for the recorded continuation scope.
