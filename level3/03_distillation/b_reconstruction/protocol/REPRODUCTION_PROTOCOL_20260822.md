# On-device distillation — independent reproduction protocol

**Written before any student was fitted.** 2026-08-22.

## Why this exists

The 2026-08-16 on-device distillation (S7 Figure 5, and one sentence of the abstract) is a NEW
model fit. Its training code, pre-registration record, seeds and per-fold predictions were written
to a session-scoped scratchpad that no longer exists. Only the aggregate record
`analysis_addon/results/distillation_ondevice_20260816.json` survives. See
`S7_NUMBERS_AUDIT_20260822.md` finding F1.

This run therefore **cannot verify** the original numbers in the strict sense: without the original
seeds and code, an exact bit-level reproduction is impossible in principle. What it can do is test
whether the reported result is *reconstructible* — whether an independent implementation of the
documented protocol, on the same frozen data partition, lands in the same place.

## What this run is, and is not

- It **is** an independent reconstruction from the protocol as documented in the manuscript
  Methods and the S7 revision report.
- It **is not** the original pre-registered run, and it must never be described as one.
- It **does not** authorize changing any number in the manuscript. The manuscript's values stay as
  they are unless the accountable authors decide otherwise after seeing this result.
- The original pre-registration cannot be recovered, so the manuscript's use of the word
  "pre-registered" is not settled by this run either way.

## Confirmation-fold status

The original protocol declared fold 1 "touched exactly once". This run evaluates fold 1 again.
That is recorded here explicitly rather than glossed: this is a **second, post-hoc touch of the
confirmation fold by a different implementation**, performed for audit purposes at the accountable
author's instruction. It is not a new confirmatory test and no selection, tuning or architecture
choice in this run may be informed by fold-1 results. All design choices below are fixed in this
document before the first fit.

## Frozen design (fixed before fitting)

Data layer is reused verbatim from the frozen 2026-08-13 contract
(`DISTILLATION_CONFIG_20260813.json`, input SHA-256 gate enforced):

- development n = 11,436; 30%-identity groups = 10,715; outer folds 0–6, group-exclusive.
- confirmation fold = 1 (1,617 rows / 1,531 groups). Training folds = {0, 2, 3, 4, 5, 6}.
- B1 = the contract's `B1_25_TRANSPARENT_PLUS_400_ORDERED_DIMERS` block, logistic C=1000,
  lbfgs, max_iter 20000, standardized. This is the canonical, reproducible B1.
- Teacher = plain logistic regression on frozen ProtT5 embeddings (same solver settings), as the
  manuscript states for the on-device run — not the Figure 4 PLS pipeline.
- Teacher soft targets for training rows are cross-fitted **within the training folds only**
  (6-way leave-one-fold-out over {0,2,3,4,5,6}), so no student row ever sees an in-sample teacher
  response. Cross-fit slices come from `teacher_crossfit.make_crossfit_plan`.

Student (reconstructed from the documented description):

- learned residue embedding over the canonical 20-letter alphabet plus padding;
- three parallel 1-D convolution kernels (widths 3, 5, 7);
- global max- and mean-pooling over each kernel's channels;
- concatenated with the 25 transparent descriptors through a small projection;
- a penultimate representation, then a single logit;
- teacher-free at inference; NumPy-only forward path re-implemented and checked against torch.

Loss = hard-label BCE + soft-label KD (temperature 4, weight 0.5) + feature distillation
regressing the penultimate representation onto a fold-local low-rank projection of the ProtT5
embedding. The hard-only control is the identical architecture and seed with both distillation
terms disabled.

Training: AdamW, lr 3e-3, weight decay 1e-4, batch 128, gradient clip 1.0, 40 epochs, CPU, single
thread, seed 20260813. No early stopping, no checkpoint selection, no hyperparameter search.

Uncertainty: paired cluster bootstrap over fold-1 `cluster_id`, 10,000 replicates, seed 20260813.

## Declared comparison criterion (fixed before fitting)

The reconstruction is reported as:

- **CONSISTENT** if the distilled student's fold-1 ROC-AUC is within 0.02 of the reported 0.730
  AND both reported contrasts (over hard-only, over B1) keep the same sign with lower bounds above
  zero.
- **PARTIALLY CONSISTENT** if the AUC is within 0.02 but a contrast loses significance, or the AUC
  is within 0.04 with both contrasts still positive.
- **NOT REPRODUCED** otherwise.

Whatever the outcome, it is reported as-is. No threshold in this section may be revised after
seeing a result, and no architecture or training setting may be retried to move a result toward
the reported value. A single fit per arm is run.

## Known deviation to report either way

The reported on-device record uses B1 = 0.701. The canonical contract B1, refit here on the same
training folds, is 0.7069 — the same value the 2026-08-13 audit recorded (0.707059). The
reproduction therefore reports the student's margin against **both** the reported 0.701 and the
canonical 0.7069, and does not adopt the lower baseline.

## Implementation constants fixed before fitting (amendment, same day, pre-fit)

These were not pinned by the surviving documentation, so they are declared here before the first
student is fitted rather than chosen afterwards.

- **Sequence window** `MAX_LEN = 768`, symmetric head+tail truncation for longer chains. 94.1% of
  development sequences are unaffected (median 318, p90 646, max 2825). Both termini are kept
  because signal peptides and C-terminal tags both matter for this endpoint.
- **Loss** `(1 - alpha) * BCE(hard) + alpha * T^2 * KL(student || teacher) + beta * MSE(feature)`
  with `alpha = 0.5`, `T = 4`, `beta = 0.1`.
- **Feature-distillation target**: fold-local PCA to 32 components of the frozen ProtT5 matrix,
  fitted on the training folds only and standardized. Never fitted on the confirmation fold.
- **Threads**: training uses 8 torch threads. The single-thread constraint applies only to the
  reported inference-latency benchmark, as in the original record.
- **One fit per arm**, seed 20260813. No repeats, no seed sweeps, no restarts.
- **Reconstructed parameter count**: 99,617 deployable parameters against the reported 100,719
  (1.1% apart). The architecture was **not** tuned to match the reported count; the difference is
  reported as-is and is itself evidence of independent reconstruction.
