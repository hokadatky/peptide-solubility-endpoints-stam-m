# Prespecified plan — screening-budget re-expression of the retrospective external evaluation

Run id: `stam_external_screening_budget_20261025`
Written: 2026-10-25, **before** any statistic in this directory was computed.
Parent evidence root: `../stam_nesg_external_20261016/` (untouched by this run).

## 1. Why this analysis exists

The manuscript's Discussion states that a ROC-AUC difference

> does not directly quantify success within that fixed top-$k$ budget, for which the number of
> successes among the selected candidates and their precision are the relevant quantities

and then declines to give any such number. The statement is correct but leaves the reader without
the quantity the paper itself names as decision-relevant. This run supplies it for the external
evaluation that has already been reported.

## 2. What this analysis is, and is not

**It is a re-expression of an existing result, not a new comparison.** No model is fitted,
refitted, reselected or recalibrated. The arms, their hyperparameters, their calibration maps and
their row-level external predictions were all fixed and sealed in the parent run on 2026-10-16 and
are consumed here read-only. The only thing that changes is the summary statistic applied to those
frozen predictions: top-$k$ precision instead of ROC-AUC.

**Honest label: descriptive, retrospective.** It is not confirmatory and cannot become so. The
external outcomes had already been examined during earlier method development, and the exploratory
arms were specified after the first external result was seen. A different summary statistic does
not reset either fact. Nothing computed here may be reported as external validation, as evidence
of external benefit, or as a confirmatory comparison.

## 3. Inputs (read-only, hash-checked)

| Purpose | File |
|---|---|
| Row-level external labels; primary two arms, raw and calibrated | `../stam_nesg_external_20261016/private/external_predictions.npz` |
| Row-level scores for the five exploratory arms, raw and calibrated | `../stam_nesg_external_20261016/exploratory_improvement/predictions_sealed.npz` |
| Stored ROC-AUC / contrast values used as join and reimplementation checks | `../stam_nesg_external_20261016/results/external_retrospective.json`, `../stam_nesg_external_20261016/exploratory_improvement/results.json` |

The two prediction files are joined on `record_id`. The join is only accepted if `record_id` is
identical and in identical order in both files; otherwise the run aborts.

## 4. Prespecified quantities

Pool: all 3,100 external records, 2,723 sequence-identity clusters, 1,550 soluble and 1,550
insoluble. Prevalence of the pool is therefore 0.5 by construction.

Probability scale: the **development-calibrated** probabilities, matching
`probability_type = "development-calibrated"` in the parent primary contrast. Raw scores are
computed as well and any ranking difference is reported.

Budget grid, fixed before computation and not to be changed afterwards:

    k ∈ {24, 48, 96, 192, 384, 1550}

The first five are standard microplate formats, i.e. the sizes in which an expression screen is
actually run. The sixth equals the number of positives in the pool, so precision at that budget is
R-precision, a standard rank summary. No other k may be added after the results are seen, and no k
may be dropped.

Per arm and per k:

- `n_success`: number of soluble outcomes among the k highest-scoring records
- `precision`: `n_success / k`
- `lift`: `precision / 0.5`, the pool prevalence

Prespecified pairings, matching the contrast structure already reported in the parent run:

| contrast | reference | candidate | status in the paper |
|---|---|---|---|
| primary | `B1_ORDER_LOGIT` | `B1_ORDER_PLUS_PROTT5_PLS_LOGIT` | prespecified in the parent run |
| exploratory 1 | `B1_LOGIT` | `FULL_LOGIT` | exploratory in the parent run |
| exploratory 2 | `B1_LOGIT` | `PCA_LOGIT` | exploratory in the parent run |
| exploratory 3 | `B1_HGB` | `PCA_HGB` | exploratory in the parent run |

`B1_ORDER_LOGIT` and `B1_LOGIT` are asserted to be the same fitted model (identical stored external
ROC-AUC and identical row-level scores); the run aborts if they are not.

## 5. Uncertainty

Paired cluster bootstrap on the difference in top-$k$ precision, replicating the estimator already
used in the parent run (`vendor/analysis_addon/src/clustered_stats.py::paired_cluster_bootstrap`)
exactly: clusters resampled with replacement to the original cluster count, all rows of a sampled
cluster retained, replicates in which the resampled pool contains only one outcome class discarded
as invalid, percentile intervals. 10,000 replicates, seed 20261016.

The reimplementation is validated by reproducing the parent run's stored ROC-AUC contrast,
`point_gain = 0.0034730489073881987` with `ci95 = [-0.012…, +0.019…]`, to within floating-point
equality before any top-$k$ interval is accepted. If it does not reproduce, the run aborts.

Intervals condition on the fitted predictions and exclude refitting, historical
method-selection and between-study uncertainty, as in the parent run. Because cluster resampling
changes the realized pool size, the realized row count per replicate is recorded; `k` is an
absolute budget, not a fraction of the pool.

Every arm × k cell and every contrast × k cell is reported, favorable or not. The count of cells
shown is stated so that no selective presentation is possible. Multiplicity: unadjusted intervals
are reported and labeled as such; this is a descriptive re-expression of an already-reported
comparison, so no adjusted interval is presented as inferential.

## 6. Interpretation limits that must be carried into any text written from this run

1. The pool is balanced 1,550/1,550 **by the source publication's construction**. A real screening
   pool is not balanced, so `precision` here is not a laboratory success rate and `lift` is not an
   expected improvement in a real campaign. These are properties of this balanced benchmark pool.
2. Discrimination on this set is weak for every arm (ROC-AUC 0.537–0.587). Top-$k$ precision
   inherits that weakness; a value near 0.5 means the ranking is close to uninformative at that
   budget.
3. The clusters are 87% singletons, so cluster resampling is close to record resampling here, and
   the intervals do not capture between-study variation.
4. Nothing here licenses a statement about direct aqueous peptide solubility.

## 7. Outputs

- `screening_budget.py` — the single deterministic script, no network, no fitting
- `results.json` — every computed quantity, with input hashes and environment record
- `metrics.tsv` — the arm × k table
- `contrasts.tsv` — the contrast × k table with intervals
- `RESULTS_JA.md` — what was found, in Japanese, including the unfavorable cells
- `SHA256SUMS.txt`
