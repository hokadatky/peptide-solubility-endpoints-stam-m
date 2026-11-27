# Retrospective analysis freeze — 2026-08-10

Status: **retrospective, data-informed analysis freeze**.

This freeze records an already-viewed and already-developed worked-example audit. It is not a preregistration and does not establish a locked external benchmark.

## Author authorization

- Governance: retrospective reanalysis of the recombinant soluble-expression worked example is authorized.
- Manuscript integration: prohibited until author review.
- Author-decision SHA-256: `c5f9399748500aaf5209970ccbf913edda1dd7f339c6d9aaa499faff8b13ef6a`.

## Primary contract

- Endpoint: recombinant soluble-expression status in *E. coli*; not direct aqueous solubility.
- Reference: B1_ORDER_LOGIT (25 transparent AAC/physicochemical variables + 400 ordered dimers).
- Candidate: B1_ORDER_PLUS_PROTT5_PLS_LOGIT (fold-local supervised ProtT5 PLS appended to B1).
- Learner: nested L2 logistic regression, lbfgs, max_iter=20000, class_weight=None.
- C grid: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]`.
- PLS component grid: `[1, 2, 3, 5, 8, 16]`.
- Outer split: fixed existing seven-fold `development_fold` assignment; grouped and retrospective.
- Inner split: `v4-sgkf-1.2.0-seeded-token-shuffle-off-gates-v2`; five folds with fail-closed balance gates.
- Primary uncertainty: paired cluster bootstrap, 10,000 replicates, seed 20260809.
- Primary margin: AUC gain 0.02; log10 RMSE gain 0.10; Spearman gain 0.05.
- External procedure: fit candidates on `fit`, select on `model_selection`, refit on their union, fit Platt mapping on `calibration`, seal, then apply once retrospectively.

## Prohibited changes

- No manuscript, caption, cover letter, existing audit JSON, or existing figure modification.
- No ESM-2 completion, amyloid, learning curve, LoRA, attention pooling, synthesis performance, or app work before the author checkpoint.
- No external-label-driven tuning, threshold selection, feature selection, or model replacement.
- No row-level data, sequences, or record identifiers in tracked outputs.

## Provenance

- Config SHA-256: `f5474cb07f86cb6a7dc8d789a8a8cb2a062df740a2901314ddcb62a2423dfe60`.
- Code/config source-tree SHA-256: `37a743ee5d1cb6aa857f69fb36bf2014319c45d18db2cba1d61427b74d2f02cc`.
- Author decision SHA-256: `c5f9399748500aaf5209970ccbf913edda1dd7f339c6d9aaa499faff8b13ef6a`.
- Input canary hashes:
  - `development_features`: `74ee352dfeff9be1b2500261ab6d60fb2aed8870fb280b5b300e8f4ebd4712b9`
  - `embedding_index`: `0988ba5f71b8da7c9af65d5d05baa0616538a9629ac88c80e9f036430c22bf94`
  - `external_clustered`: `6053e4d90e26ab2a13dd7e6f4c99da66e269a4a2711e5645ee13b70594e3b469`
  - `external_features`: `a906652cd246931dfbf1c75e4b30cf8fc59586ebee7fe77adae3bf114e0c698c`
  - `feature_manifest`: `be7638fbc9bdf421661710ef8319a2be58473f4de4bfe351edbd9adda3279f50`
  - `prott5_array`: `be0a480acaf19f2deada752e5befb7d568f90b73f089a2ec6b74aa75ce7cdf45`
  - `train_partitions`: `28a63f9d40c47bda0ff936830eacbf818572b78f67f23417ad01e3bee528ad0e`

Any contract change requires an append-only amendment and a new run_id; this run must not be overwritten.
