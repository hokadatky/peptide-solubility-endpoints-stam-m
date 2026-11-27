# Analysis amendments

## 2026-08-09 — v2 before model fitting

- The initial `addon_20260809_v1` Phase 0 preflight created a valid author/input
  freeze before the full analysis source tree had finished implementation.
- No model fit, OOF prediction, bootstrap, external evaluation, reclustering,
  embedding generation, figure generation, or manuscript change occurred in
  v1.
- After the synthetic harness was completed and corrected, the source-tree
  hash changed. To avoid overwriting the v1 freeze or run artifacts, the
  executable analysis was amended to `addon_20260809_v2` with a separate
  private run root and `RETROSPECTIVE_ANALYSIS_FREEZE_20260809_V2.md`.
- Scientific contract, author decisions, input canaries, margins, endpoint,
  and primary contrast are unchanged.

## 2026-08-09 — v3 recovery after interrupted v2 secondary arm

- The v2 `internal-primary` process remained alive after handoff and was
  verified read-only as actively consuming approximately five CPU cores in the
  high-dimensional raw-fusion secondary arm. It had run for more than four
  hours without producing that arm's artifact.
- Before termination, v2 had completed and immutably saved the historical
  bridge plus OOF predictions for `B0_TRANSPARENT_LOGIT`, `B1_ORDER_LOGIT`,
  `B0_PLUS_PROTT5_PLS_LOGIT`, and `PROTT5_PLS_ONLY_LOGIT`. It had not produced
  the raw-fusion arm, the primary candidate, the internal result/manifest, or
  any external/shift/modified/release result.
- PID 57761 was terminated with SIGTERM only after confirming its exact command,
  elapsed time, active CPU use, and the absence of a raw-fusion checkpoint. No
  file was removed or overwritten.
- Recovery uses a new run ID and private root (`addon_20260809_v3`). The v2 run
  remains immutable. Imported v2 artifacts must pass row alignment, fold,
  finiteness, metric, and SHA-256 checks; imports are recorded in a recovery
  manifest.
- The scientific primary contract, endpoint, inputs, seven fixed outer folds,
  inner grid, margins, bootstrap count/seed, and external procedure are
  unchanged. Primary reference and candidate are recomputed under v3 and the
  reference prediction is compared numerically with the v2 checkpoint.
- The high-dimensional raw-fusion arm is deferred for computational
  tractability and cannot support a claim. To avoid a less conservative
  multiplicity family, its planned cluster-swap p-value is represented as 1.0
  when Holm adjustment is calculated across the original four secondary arms.
- Execution-only safeguards are added: primary-first arm order, arm/seed-level
  immutable checkpoints, resume validation, and a vectorized implementation of
  the same cluster-wise swap randomization. These changes do not alter the
  estimand or model contract.
- PLS inner selection fits the declared maximum component count once per inner
  fold and slices its sequential score-coordinate prefixes for the smaller
  declared values. A pre-fit dry run found maximum absolute differences below
  `6e-15` versus fitting each component count separately; k-specific matrix
  scaling, logistic fits, grids, tie-breaking, and final selected-k refits stay
  unchanged. This is an execution-only cache, not a model change.
- The training-label negative control uses 50 frozen-hyperparameter refits. For
  each outer fold and repeat, only that fold's training labels are permuted;
  test labels are never shuffled, and the supervised PLS projection is refit
  on the permuted training labels. The predeclared diagnostic requires 0.5 to
  lie within both model-AUC null 95% percentile ranges and zero within the
  candidate-minus-reference gain range. Row-level null predictions remain in
  private immutable checkpoints.

## 2026-08-10 — v4 after grouped-split invalidation

- V2/V3 nonhistorical predictions and checkpoints are invalidated as recorded
  in `V3_INVALIDATION_20260810.md`; their viewed AUC values cannot guide V4.
- The primary endpoint, contrast, metric, grids, seeds, practical margin, and
  learner objective are unchanged.
- The grouped allocator is replaced by version-pinned SGKF with seed-hashed
  group tie order, `shuffle=False`, canonical fold labels, and fail-closed 5%
  row/class and 0.02 prevalence gates. Forty-nine production assignments passed
  before model fitting.
- Logistic `max_iter` rises from 5,000 to 20,000 without changing solver, tol,
  scaling, or objective. Nonconverged inner candidates are ineligible; a
  nonconverged outer refit is a hard stop. PLS and probability diagnostics are
  also mandatory.
- Checkpoint schema v2 binds feature content, labels, groups, exact outer and
  inner assignments, numerical policy, selected specification, convergence
  diagnostics, predictions, and an artifact digest. V1 checkpoints fail closed.
- All five affected non-raw arms are recomputed. Raw fusion remains deferred,
  enters the frozen four-member Holm family with p=1 only for multiplicity, and
  is explicitly not evidence of a null effect.
- External evaluation is development-only until sealing: fit candidates on
  `fit`, select on `model_selection`, refit on their union, fit Platt calibration
  on `calibration`, write an immutable seal, and only then access external labels.
- This is a retrospective, data-informed amendment, not preregistration.
