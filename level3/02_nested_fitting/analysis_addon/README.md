# Retrospective analysis add-on

This tree contains the `addon_20260809_v3` retrospective, data-informed audit
authorized in `protocol/AUTHOR_DECISIONS_20260809.md`. It is not a
preregistration and its outputs must not be integrated into the manuscript,
existing figures, or existing audit JSON without a separate author decision.

The primary SoluProt contrast is the order-aware non-pLM baseline
`B1_ORDER_LOGIT` versus `B1_ORDER_PLUS_PROTT5_PLS_LOGIT`, evaluated with the
same nested grouped harness internally and on the retrospective external set.
Primary uncertainty is paired cluster bootstrap with 10,000 replicates.

Row-level predictions and sequences remain in the private run root outside the
nested Git repository. `results/` is aggregate-only and is checked by
`tests/test_release_privacy.py`.

V3 preserves the interrupted V2 run byte-for-byte. It validates and imports
only completed V2 OOF artifacts, recomputes the primary reference/candidate,
and requires numerical parity of the recomputed reference. Long nested fits
checkpoint each outer fold. Re-running a stage reuses only checkpoints whose
full execution contract matches; a mismatch fails closed.

Execution order (from the nested Git root):

```bash
python analysis_addon/src/preflight.py --config analysis_addon/config/analysis_config_v3.yaml \
  --author-decisions analysis_addon/protocol/AUTHOR_DECISIONS_20260809.md \
  --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3
python -m pytest -q analysis_addon/tests
python analysis_addon/src/recover_v2.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3
python analysis_addon/src/run_soluprot_reanalysis.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3 --stage internal-primary
python analysis_addon/src/run_soluprot_reanalysis.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3 --stage external-retrospective
python analysis_addon/src/run_soluprot_shift_audit.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3
python analysis_addon/src/run_modified_peptide_audit.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3 --stage collision-and-family-feasibility
python analysis_addon/src/aggregate_release.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3 --output-dir analysis_addon/results
python -m pytest -q analysis_addon/tests/test_release_privacy.py
python analysis_addon/src/privacy_reproducibility_audit.py --config analysis_addon/config/analysis_config_v3.yaml --run-dir ${WORKSPACE}/paper/general_peptide_solubility_scoping/analysis_addon_run_20260809_v3 --release-dir analysis_addon/results
```

The private independence denylist must be placed at
`<run-root>/privacy/private_denylist.txt` from the already-authorized handoff
before the final privacy audit. Do not copy its tokens into tracked files.
