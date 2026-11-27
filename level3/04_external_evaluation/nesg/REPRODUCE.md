# Reproduce the additional NESG evaluation

This is a local STAM-M retrospective external evaluation, not an app release. The manuscript and previous results are unchanged. Read protocol/ANALYSIS_PLAN.md first.

## Inspect or verify the completed run

Use the original sc8 environment (Python 3.10.17, NumPy 1.26.4, scikit-learn 1.2.0, SciPy 1.15.3, pandas 2.3.3). No scientific package was upgraded for this evaluation.

From the workspace root:

```sh
python -B paper/stam_nesg_external_20261016/src/verify_results.py
```

Verification recalculates AUC, average precision, Brier and log loss from the private saved predictions, then independently implements weighted cluster bootstrap and compares all 10,000-replicate 95% CI endpoints. It checks prediction seals, input hashes, frozen source and the original manuscript/repository baseline. It does not refit models.

## Refit from the preserved original inputs

Do not overwrite this completed run. Create a new local reproduction directory and copy only src/, vendor/, protocol/original_config_v4.yaml and protocol/ANALYSIS_PLAN.md into it. Do not copy the completed run's resolved_config, freeze, checkpoints, private predictions or results. The script derives its output directory from its own location, so a fresh copy writes to its own new root. It uses the original-config input paths; no original input is modified. If paths change, relocate the input references explicitly and preserve the recorded input hashes and all scientific settings.

Run the fresh copy's src/run_external.py with --preflight-only, then without that flag. Use OPENBLAS_NUM_THREADS=4 and OMP_NUM_THREADS=4. The preflight validates seven input digests, package versions, record order, shapes, exact sequence overlap and group-disjoint development roles. A genuine earlier V4 internal-primary completion manifest is checked as source provenance. Do not manufacture that manifest if it is absent.

After a fresh fit, run that copy's src/make_safe_prediction_copy.py, then src/verify_results.py in the same environment. The first command preserves the original prediction archive and creates a pickle-free copy with Unicode identifiers; numerical predictions and labels are unchanged. The completed run already contains this copy, so verification alone is sufficient for the existing artifacts. These two post-analysis utilities are separate from the pre-execution fitting-code freeze.

The copied src/external_runtime.py supplies label-delayed loading and the old V4 training/prediction helpers. Its original standalone CLI is NOT the entry point: use src/run_external.py. Only import-root paths were changed in that helper. Full original training/statistics implementation is retained under vendor/analysis_addon/src/ with a pre-execution source manifest. This is a local reproduction bundle; the historical path dependencies must be made portable and rights/access conditions checked before any external distribution.

The entry point refuses to overwrite a completed result. Intermediate arm checkpoints are bound to input/configuration contracts and may be resumed unchanged. The execution seal records that this run did not parse external outcomes before sealing predictions; it does not claim the dataset had never been viewed historically.

## Retained files

- protocol/ANALYSIS_PLAN.md: fixed two-arm comparison, sampling unit, bootstrap and interpretation.
- protocol/PRE_EXECUTION_FREEZE.json: timestamp and checksums of the plan, resolved configuration and fitting code before outcome parsing.
- protocol/preflight.json: original input identity and environment checks, external labels not parsed.
- protocol/test_external_seal.log: synthetic contract tests.
- checkpoints/external/: checksummed raw/calibrated prediction checkpoints, development-selected settings and convergence diagnostics. These are prediction checkpoints, not deployable fitted-weight exports.
- manifests/external_development_only_seal.json: predictions fixed before new-run label parsing.
- private/external_predictions.npz: private aligned outcome/prediction records for local verification. Do not publish automatically.
- private/external_predictions_safe.npz: pickle-free copy used by the verifier; results/SERIALIZATION_REPAIR.json records its provenance and numerical invariance.
- results/external_retrospective.json: aggregate accuracy/calibration/paired CI results.
- results/INDEPENDENT_VERIFICATION.json: independent numerical checks and original-artifact protection.
- run.log: execution log.

Confidence intervals condition on fixed fitted predictions and external clusters. They omit refitting, historical method selection and study-to-study uncertainty. Balanced external sampling is not the prevalence of a user's future experiment. No student, smartphone implementation or direct peptide concentration model is evaluated here.
