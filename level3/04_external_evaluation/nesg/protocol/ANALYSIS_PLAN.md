# NESG retrospective external evaluation — revision TSTM-2026-0051

Authorized by the author on 2026-10-16: perform additional NESG evaluation. This supersedes the earlier no-external-execution scope for this NEW run only. Submitted manuscript, figures, earlier results and other projects remain unchanged.

## Question and interpretation
Does the B1+ProtT5 representation improve discrimination over the matched B1 logistic baseline on the existing NESG-derived SoluProt test set? This is a retrospective, data-informed, reviewer-requested external evaluation. External labels and earlier model results had historically been viewed. This new execution withholds outcome parsing until predictions are sealed; that does not erase historical exposure or establish a prospective/fully untouched test.

Scope: 3,100 external records, expected 2,723 previously assigned external 30%-identity groups. Endpoint: recombinant soluble expression in E. coli, not aqueous peptide concentration. Original SoluProt source used a low-similarity test set; this run audits input identity and exact train/test sequence overlap, does not recreate the original USEARCH global-identity filter or claim absence from pLM pretraining.

## Frozen before fitting and external outcome parsing
Use the previously specified V4 development-selection/calibration algorithm, with no change based on NESG results. Two arms only: B1_ORDER_LOGIT and B1_ORDER_PLUS_PROTT5_PLS_LOGIT. B1 has 25 AAC/physicochemical features plus 400 ordered dimers; candidate adds supervised training-local ProtT5 PLS. Encoder/checkpoint/pooling are those of the verified existing embeddings.

Development-only roles (existing group-disjoint assignment): fit 8,198 records / 7,653 groups; model_selection 1,621 / 1,531; calibration 1,617 / 1,531. Fit grid candidates on fit; choose AUC on model_selection, with the existing deterministic tie rule smaller k then smaller C. Refit on fit+model_selection (9,819 records); fit Platt calibration only on calibration. Scaling and PLS are fitted inside each eligible training set. Neither NESG covariates nor labels determine scaling, PLS, model selection or calibration.

C grid: 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000. PLS grid: 1, 2, 3, 5, 8, 16. L2 logistic lbfgs, tol=1e-4, max_iter=20000, no class weighting. Retain the existing convergence/finite-value gates. An ineligible selection candidate is logged; final refit/calibration nonconvergence is a hard stop. No post-test grid expansion, retry with alternative models, threshold search, or student redesign.

Primary estimand: calibrated-candidate AUC minus calibrated-B1 AUC, paired external cluster bootstrap, 10,000 replicates, seed 20260909 (previous external contract: 20260809+100). Resample groups with replacement and retain all rows of each sampled group. Report percentile 95% CI and valid/invalid counts. This is one planned contrast in this revision, not a correction for historical model development. CI is conditional on fitted predictions, excludes refit, method-selection and between-study uncertainty. No utility claim from the previous +0.020 design margin, no observed-power claims, no threshold success-rate claim.

Secondary descriptive metrics for both raw and development-calibrated probabilities: AUC, average precision, Brier, log loss, calibration intercept/slope and reliability bins. External calibration diagnostics are measurements only; do not use them to recalibrate predictions. Save all these metrics whether favorable or not. No comparison of external AUC with internal AUC as a paired significance test; different populations and fitting/evaluation designs.

## Execution and retention
1. Hash-check the seven original inputs and imported source code. Check current scientific package versions against the original environment.
2. Parse development labels and external covariates only, validate row/embedding order, group-role disjointness, dimensions and finiteness, and exact sequence overlap.
3. Generate two-arm external predictions using development-only selection/refit/calibration. Record each selection grid and diagnostics, checksummed checkpoints and a development-only prediction seal.
4. Only after the seal, load external labels aligned by record_id. No fitting follows this step except descriptive calibration diagnostics (which never alter prediction files).
5. Save private row-level predictions locally, aggregate metrics, uncertainty and input/code/output manifests. Independently recalculate metrics from saved predictions and verify no changes to submitted artifacts.

Run root: paper/stam_nesg_external_20261016. No old external-stage manifest is fabricated or overwritten. Existing V4 internal-primary completion is a provenance prerequisite, not a declaration that this new external run had already completed. Raw records, model inputs, predictions and identifiers are not uploaded or sent externally. No public release or manuscript change is authorized by this run.

Model-fitting environment deliberately retains original sc8 package versions for reproducibility; do not upgrade the scientific stack merely to follow a generic installation example. Keep all inputs/code and outputs local. Expected intermediates are small relative to existing data, so an external scratch volume is not needed.
