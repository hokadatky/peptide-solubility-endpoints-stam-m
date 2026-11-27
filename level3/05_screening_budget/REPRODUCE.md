# Reproducing this run

```bash
cd paper/stam_external_screening_budget_20261025
python -I screening_budget.py
```

The script is deterministic and offline. It fits nothing: the only inputs are the row-level
external predictions sealed by the parent run `../stam_nesg_external_20261016/` on 2026-10-16,
and that run's stored metric JSON files, all read-only. Expect roughly 25 minutes, almost all of
it in the 25 cluster-bootstrap passes of 10,000 replicates each.

The script aborts rather than writing partial output if any of these preconditions fails:

- `record_id` or `external_cluster_id` differ between the two prediction files
- `B1_ORDER_LOGIT` and `B1_LOGIT` are not the same fitted model
- any recomputed external ROC-AUC differs from the parent run's stored value by more than 1e-12
- the reimplemented cluster bootstrap does not reproduce the parent run's stored primary contrast
  point estimate and 95% interval to within 1e-12
- the pool is not 3,100 records with 1,550 positives

Environment used: Python 3.10.17, numpy 1.26.4, scikit-learn 1.2.0, macOS 15.4 arm64 (conda env
`sc8`). The bootstrap seed is 20261016 and the budget grid is fixed in `PLAN.md`; both were set
before execution and must not be changed to reproduce these values.

Verify integrity with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```
