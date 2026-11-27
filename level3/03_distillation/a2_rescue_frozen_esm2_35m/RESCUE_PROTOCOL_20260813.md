# Bounded exploratory distillation rescue protocol

Date: 2026-08-13  
Status: exploratory rescue; does not overwrite the frozen V1 negative result

## Question

Can a student that preserves an established B1 information path avoid the
severe overfitting of TriPepLite-v1 and retain at least 80% of the selected
ProtT5 teacher's ROC-AUC gain?

The endpoint remains recombinant soluble-expression status in *E. coli*, not
direct aqueous solubility of a purified peptide.

## Why this is exploratory

V1 confirmation fold 1 has already been opened and its result has influenced
the present architecture diagnosis. A rescue model evaluated on that fold
cannot be called a new untouched confirmation. The V1 result remains the
confirmatory record. Any rescue result is an exploratory feasibility result
requiring a future locked external or prospective evaluation before promotion.

## Fixed success calculation

For each evaluation set:

`target AUC = B1 AUC + 0.80 * (teacher AUC - B1 AUC)`.

The candidate must also outperform the corresponding hard-label fit of the
same feature architecture. Proper-scoring metrics are reported but do not
replace the ROC-AUC gate.

## Stage 1: design-fold dry run only

- fit folds: 2--6;
- evaluation fold: 0;
- teacher targets: existing leakage-controlled V1 cross-fitted ProtT5 targets;
- teacher evaluation score: the frozen V1 fold-0 score;
- maximum candidate mechanisms: three;
- no fold-1 data may be loaded by the dry-run script.

Candidate mechanisms are fixed before the dry run:

1. B1 soft-target logistic control;
2. B1 plus deterministic hashed 2--5-mer linear student;
3. B1 plus the pre-existing ESM-2 35M pooled representation, used only as a
   capacity diagnostic because its legacy pooling/truncation contract is not
   acceptable as final evidence.

Each soft fit uses target
`0.25 * hard_label + 0.75 * cross-fitted_teacher_probability`. Fractional
cross-entropy is implemented by duplicating each row with positive and negative
sample weights that sum to one. The hard control uses identical input features
and estimator settings with observed labels only.

## Stop/go gate

Stop after Stage 1 unless at least one candidate:

- meets the 80% design-fold retention target;
- exceeds its hard-only counterpart in ROC AUC; and
- has finite predictions and no group overlap.

If a candidate passes, freeze exactly one candidate by highest design-fold
ROC AUC, breaking ties by lower log loss. Only then may a separately labelled
exploratory fold-1 evaluation be run once. No post-fold-1 tuning is allowed.

## Privacy and file boundaries

Row-level predictions remain under
`paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2/`.
Only aggregate JSON and the protocol may enter this tracked plan directory.
S0, S1, S2, V4, the app and the frozen V1 run are read-only during the dry run.

