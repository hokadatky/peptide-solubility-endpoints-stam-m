# Reproducing the exploratory comparison

This directory is an additional exploratory analysis requested after the original NESG result was known. It never replaces that result. Read PLAN.md and the Japanese result report before interpreting any positive contrast.

The input datasets, features and ProtT5 embeddings are the parent's original hash-checked inputs. The original sc8 environment is retained: Python 3.10.17, numpy 1.26.4, pandas 2.3.3, scipy 1.15.3 and scikit-learn 1.2.0. No package upgrade or embedding regeneration is required. The original source CSV and FASTA files must also be present for audit_inputs.py.

From the workspace root, to verify completed models and results without fitting:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -B paper/stam_nesg_external_20261016/exploratory_improvement/verify_models.py
```

The verifier checks all input and frozen-code hashes, model selection against the complete development grid, training-only scaler/PCA sample counts and means, saved-model prediction replay, external metrics, both ordinary and multiplicity-adjusted bootstrap intervals, the preceding NESG result, and the submitted manuscript/repository baseline. Its 10000-replicate bootstrap uses explicit resampled row arrays, independently of the fitting script's weighted-rank implementation. No model is refitted during this verification.

For a full refit, create a separate sibling directory under paper/stam_nesg_external_20261016 and copy only PLAN.md, audit_inputs.py, run_models.py and verify_models.py into it. Run the copied audit_inputs.py, then run_models.py, then verify_models.py with the same environment and thread variables. Do not copy completed results, freezes, checkpoints, selection records or fitted models. The sibling uses the same parent's vendor/src and original configuration; its output root is derived from its own file location. This is a local reproduction procedure; historical absolute input paths and distribution permissions need separate work before a public portable package can be claimed.

An interrupted original execution can resume unchanged after confirming its freeze and checkpoints. A completed results.json prevents rerunning the fitting entry point in place. Do not modify the frozen plan or fitting source to add more models after outcomes have been inspected.

Retained evidence:

- INPUT_AUDIT.json: source-label/sequence identity, all primary features independently recomputed, embedding provenance, fixed length/tag diagnostics.
- FREEZE.json: pre-fit plan, code, environment, and original parent-evidence hashes.
- selection/: every attempted development hyperparameter setting, eligibility, convergence, chosen setting and held-back calibration diagnostics.
- models/: five selected fitted pipelines including training-local preprocessing and development-only calibration. These are trusted local joblib files; they are not an app implementation or public release.
- checkpoints/: numerical selected-arm predictions and calibration-partition records.
- predictions_sealed.npz and PREDICTION_SEAL.json: all five external predictions fixed before this execution's external scoring; historical and preceding-run outcome exposure remains explicit.
- results.json, metrics.tsv, VERIFICATION.json: results, descriptive metrics and independent verification.
- run.log: fitting log.

The original failed-to-confirm PLS comparison, all three new comparisons, and any new favorable result must be reported together. A separate endpoint-compatible cohort remains necessary for confirmatory evaluation of a model chosen after NESG exposure.
