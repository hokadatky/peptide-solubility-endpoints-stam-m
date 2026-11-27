# Bounded exploratory improvement after NESG outcome exposure

Author instruction: investigate ways to demonstrate a genuine ProtT5 increment on external data. This authorizes a separate exploratory analysis, not replacement of the completed two-arm result. Manuscript and submitted evidence remain unchanged.

## Motivation and audit

The preceding retrospective NESG evaluation gave B1 AUC 0.536781 versus B1+PLS-ProtT5 0.540254. Both selected C=10000 and the candidate selected the largest permitted PLS dimension, 16. This motivates checking information loss and model form rather than assuming the pretrained representation lacks transferable information. First verify original label/sequence joins, feature recomputation, embedding provenance, and describe length/tag shifts. Do not remove observations or reverse labels based on predictive performance.

## Fixed exploratory family (before new fits)

Three comparisons only, with complete results reported:

1. Full embedding: standardized L2 logistic B1 versus B1 plus all 1024 ProtT5 dimensions.
2. Unsupervised compression: the same B1 logistic versus B1 plus train-local, standardized-embedding PCA with 64, 128 or 256 components, then joint standardization and L2 logistic. PCA uses full deterministic SVD, no whitening.
3. Nonlinear head: histogram gradient boosting on B1 versus B1 plus the same train-local PCA with 64 or 128 components. Both arms use the same tree grid: max_leaf_nodes 15 or 31; min_samples_leaf 20, 50 or 100; l2_regularization 1 or 10; learning_rate 0.05; max_iter 300; early_stopping=False; random_state=20261016.

For both logistic comparisons use the original C grid: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000. Max_iter=20000, tol=1e-4, lbfgs, no class weights. The common B1 logistic is fitted once. The original PLS result remains a separately retained reference, not part of the three new contrasts.

Preserve development group-disjoint roles: fit 8198, model selection 1621, calibration 1617. Select by model-selection AUC alone. Exact ties prefer smaller dimension, stronger regularization and simpler trees. Refit selected specification on fit plus selection (9819); fit Platt calibration on the reserved 1617. Log calibration-partition discrimination as an additional development diagnostic, without using it to select models. All scaling/PCA fit on the relevant training rows only. NESG covariates do not fit any transforms. No outcome-dependent subgroups, pruning, sign flips, endpoint changes or target-domain retraining.

Fit all five selected arms and seal their external predictions before loading external outcomes in the new execution. No exploratory external metrics are inspected until all arms are sealed. Do not add more recipes after seeing these results. Report all three deltas and all arm metrics, with 10000 paired external-cluster bootstrap samples, seed 20261016. Show ordinary 95% intervals and Bonferroni 98.3333% intervals for the three comparisons; the latter account for these three simultaneous comparisons only, not historical adaptation. Also give 95% individual-arm AUC intervals and descriptive AP, Brier and log loss. Intervals condition on fitted predictions and do not include refitting or historical model-selection uncertainty.

This is explicitly post-result exploratory model development. An apparently positive NESG increment does not become confirmatory by freezing this new script, calibrating only on development data, or applying multiplicity correction. A model intended for a stronger generalization claim must be fixed and evaluated in a genuinely separate, endpoint-compatible cohort after overlap/provenance review. Do not create a supposedly untouched cohort by splitting previously viewed NESG rows.

## Persistence and software

Use the original sc8 scientific stack and original hash-checked inputs. No package upgrades, original evidence changes, manuscript changes, external uploads or other-project transfers. Store selected fitted pipelines, calibration objects, all selection records, predictions, code/plan hashes, convergence and verification results here. Expected artifacts are small; no heavy embedding extraction or new large cache is planned. Preserve the completed parent run untouched.

Primary methodological context: Thumuluri et al., NetSolP (2022), https://doi.org/10.1093/bioinformatics/btab801, describes mean-pooled pretrained embeddings with classification heads and concerns about TargetTrack label/tag artifacts. These motivate auditing the local data; they do not establish the cause of this run's low external AUC. Different NESG cohort definitions in that study prevent direct performance comparison.
