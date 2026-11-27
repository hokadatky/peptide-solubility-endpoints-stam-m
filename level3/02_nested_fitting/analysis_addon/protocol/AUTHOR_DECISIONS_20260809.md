# Author decisions required before analysis execution

Date: 2026-08-09 (Asia/Tokyo)
Status: **APPROVED FOR RETROSPECTIVE REANALYSIS — MANUSCRIPT INTEGRATION NOT AUTHORIZED**

This file records the completed author decisions required by the execution plan
before model fitting, performance recalculation, bootstrap, reclustering, or
external evaluation.

The parent workspace currently states that no new pLM predictor should be fit,
whereas the additional-analysis execution plan proposes a retrospective audit
of the existing recombinant-expression worked example. The decision and its
rationale are recorded below.

## Governance resolution

Select exactly one:

- [x] Permit retrospective reanalysis of the recombinant-expression worked example.
- [ ] Do not reanalyse the worked example; permit direct-endpoint audit only.

Author decision: Permit the retrospective, data-informed reanalysis described
in `CODEX_ADDITIONAL_ANALYSIS_EXECUTION_PLAN_20260809.md`.

Rationale: The existing worked example may be audited to correct its evaluation
harness and quantify its limits. This is narrow authorization for retrospective
reanalysis; it does not authorize fitting or presenting a general aqueous-
solubility predictor, restoring superseded manuscript branches, or integrating
results into the manuscript before author review.

## Manuscript integration

Recommended choice: results remain out of the manuscript until the author has
reviewed the complete analysis package.

- [x] No manuscript integration before author review (recommended).
- [ ] Other author-approved integration rule (not selected).

Author decision: Keep all additional-analysis results outside the manuscript,
existing figures and existing audit artifacts until the author has reviewed the
complete analysis package and explicitly approves any integration.

## Practical margins

The following are the execution-plan recommendations. Change only with a
scientific rationale; append the rationale and date below.

- AUC gain margin: `0.02`
- log10 RMSE gain margin: `0.10`
- Spearman gain margin: `0.05`

Author-approved values:

- `auc_gain_delta = 0.02`
- `log10_rmse_gain_delta = 0.10`
- `spearman_gain_delta = 0.05`

Scientific rationale for any deviation from the recommendations: No deviation.
The values are approved before the new model runs as practical decision margins
for this retrospective audit. In particular, 0.10 log10 RMSE corresponds to an
approximately 1.26-fold concentration difference. Statistical evidence against
zero and evidence beyond these practical margins must remain separate decisions.

## Primary model contract

The execution-plan default contract is:

- encoder: `ProtT5`
- learner: `L2 logistic regression`
- sensitivity learner: fixed `HistGradientBoosting`

Confirm or amend each item:

- encoder: `ProtT5`
- primary learner: `L2 logistic regression`
- sensitivity learner: fixed `HistGradientBoosting`

## Authorization record

- Author name: Hiroyuki Okada
- Date/time (Asia/Tokyo): 2026-08-09 18:19 JST
- Signature or explicit written approval: The author explicitly instructed
  Codex in the current conversation to complete the four approval items in this
  template. This records that written authorization; it is not a handwritten or
  cryptographic signature.

All four required decision sections are now complete. The Phase 0 author-input
STOP condition is resolved, subject to the independent preflight and test gates
in the execution plan. This file must not be treated as a preregistration; the
correct scope label is `retrospective, data-informed analysis freeze`.
