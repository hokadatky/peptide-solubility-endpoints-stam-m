# On-device distillation — reconstruction result

**Verdict against the pre-declared criterion: `NOT REPRODUCED`.**

Run date 2026-08-22. Protocol `protocol/REPRODUCTION_PROTOCOL_20260822.md`, written before any
student was fitted. One fit per arm, no retries, no seed sweep, no tuning. The result below is
reported exactly as it came out.

## Numbers

| Arm | Reconstruction | Reported 2026-08-16 | Difference |
|---|---|---|---|
| ProtT5 teacher (plain logistic) | **0.7788** | 0.780 | −0.001 |
| B1 transparent (canonical contract) | **0.7069** | 0.701 | +0.006 |
| Hard-only student | **0.6633** | 0.708 | **−0.045** |
| Distilled student | **0.6718** | 0.730 | **−0.058** |

| Contrast | Reconstruction | Reported |
|---|---|---|
| distilled − hard-only | +0.0084, 95% CI [−0.020, +0.038] (includes zero) | +0.022, [0.005, 0.038] |
| distilled − B1 | **−0.035, 95% CI [−0.060, −0.010]** (significantly worse) | +0.029, [0.011, 0.047] |
| teacher-gain retention | **−0.49** | +0.37 |

Confirmation fold 1: 1,617 rows / 1,531 groups — identical to the frozen contract and to the
2026-08-13 audit record. Cluster bootstrap 10,000 replicates, all valid.

## What this does and does not establish

**It does not show the reported numbers are wrong.** The reconstruction is an independent
implementation of an underdetermined specification. The surviving documentation fixes the design
(folds, teacher, cross-fitting, temperature, soft-label weight, "1-D CNN over a learned residue
embedding plus transparent descriptors", feature distillation onto a low-rank ProtT5 projection)
but not the channel widths, pooling scheme, descriptor scaling, feature-loss weight, sequence
window or optimizer schedule. Any of those can move ROC-AUC by more than the gap observed here.

**The most informative signal is the hard-only arm.** It carries no distillation subtlety at all —
it is just this architecture trained on hard labels. It came out at 0.663 against a reported 0.708.
That 0.045 gap sits upstream of the distillation question and points at a difference between this
reconstructed architecture/recipe and the original one, not at the distillation claim itself.

Both reconstructed students land **below** the transparent B1 baseline (0.707), which is the same
failure mode the 2026-08-13 TriPepLite audit reported (0.641 / 0.652, retention −0.668). The
training log shows the pathology directly: the hard-only arm drove training loss to 0.033 while
scoring 0.663 on held-out groups, i.e. it overfits, and the distilled arm's loss plateaued near
5.58 under the `alpha * T^2` weighting (T = 4 gives a 16x multiplier on the soft-label term). A
mis-scaled soft-label term in **this** implementation is a live possibility. The protocol forbids
retrying with a different weight to chase the reported value, and no retry was made.

**What it does establish, positively:**

- The frozen data layer reproduces exactly (1,617 / 1,531), so the partition is not in question.
- The ProtT5 plain-logistic teacher reproduces to 0.001 (0.7788 vs 0.780).
- The teacher-free NumPy inference path works: max absolute probability difference against torch
  is 6.9e-07 over the probe batch. A sub-0.4 MB, dependency-free on-device forward pass is real;
  the deployable reconstruction is 99,617 parameters against the reported 100,719.
- **A baseline discrepancy in the reported record.** The canonical contract B1, refit on the same
  training folds, is 0.7069 — matching the 2026-08-13 audit's 0.707059 to five decimals. The
  on-device record used 0.701. The reported +0.029 margin is measured against a baseline 0.006
  below the harness's own canonical B1; against 0.7069 the reported margin would be +0.023.

## Consequence for finding F1

F1 is not closed, and this run shows why it cannot be closed by reconstruction. Where an aggregate
record is the only surviving artifact and the specification underdetermines the implementation, an
independent rebuild cannot distinguish "the original result was fragile" from "the rebuild differs
somewhere that matters". The reported values remain unverifiable from what exists in the repository.

The manuscript numbers were not changed by this run and must not be. The open decision for the
accountable authors is unchanged in kind but better informed:

1. Recover or re-author the original implementation to the point where it reproduces 0.730 — noting
   that this would again be a post-hoc touch of the confirmation fold and cannot restore the lost
   pre-registration; or
2. downgrade the Figure 5 / abstract on-device claim to an exploratory result whose audit trail is
   aggregate-only, and drop the word "pre-registered"; or
3. remove Figure 5 from the main text.

Separately, and independently of which option is chosen, the B1 = 0.701 versus canonical 0.7069
discrepancy should be resolved, because it affects the reported margin.

## Artifacts

- `protocol/REPRODUCTION_PROTOCOL_20260822.md` — pre-fit protocol and criterion
- `src/_harness.py`, `src/student.py`, `src/run_reproduction.py` — reconstruction code, now
  permanent in the repository
- `results/ONDEVICE_REPRODUCTION_20260822.json` — machine-readable result
- `results/distilled_student_state.npz` — fitted student weights
- `results/run_20260822.log` — full training log (3,066 s wall clock)
