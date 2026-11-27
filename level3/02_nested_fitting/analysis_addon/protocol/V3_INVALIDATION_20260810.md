# V3 invalidation record — 2026-08-10

Status: **V3 nonhistorical model evidence is invalid and quarantined**.

The custom grouped allocator used by V2/V3 produced severely unbalanced inner
and stability folds, including folds with only three rows. This changes the
tuning procedure and invalidates every affected selected hyperparameter,
prediction, confidence interval, stability result, and multiplicity result.
It is not evidence of direct outer-test leakage.

## Immutable source inventory

- Run root: `paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3`
- Files: 67
- NPZ checkpoints: 52 (14 primary and 38 incomplete/invalid stability checkpoints)
- Content-addressed inventory digest: `8f70c863856e0f23975a34c15e7f361cd61de3e98c054a58eecf5317361c6fc2`
- The stopped process no longer exists; no V3 file was deleted or overwritten.

## Viewed but invalid diagnostics

The invalid V3 primary OOF files give reference AUC 0.7150676343, candidate AUC
0.7865875464, and gain 0.0715199120. These values are recorded only to disclose
outcome knowledge. They are prohibited from scientific claims, manuscript
integration, model selection, confidence intervals, or V4 checkpoint reuse.

V2 secondary OOF predictions are invalid for the same broken inner allocator
and must be recomputed. All V2/V3 checkpoint schemas are rejected by V4.

## Unaffected evidence

The historical bridge uses fixed `GroupKFold` and does not call the broken
allocator. Its descriptive result remains conditionally reusable only after
code-path, group exclusivity, fold-local PLS, exact-once OOF, input hash, and
artifact-parity checks are revalidated under V4 provenance.

## Replacement contract

V4 uses run ID `addon_20260810_v4`, split contract
`v4-sgkf-1.2.0-seeded-token-shuffle-off-gates-v2`, checkpoint schema v2,
explicit convergence gates, and a sealed development-only external procedure.
All five affected non-raw arms, five stability seeds, and 50 label-permutation
controls are recomputed. Raw fusion remains a predeclared compute-based deferment
and cannot support an effect claim.
