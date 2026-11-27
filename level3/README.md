# Level 3: refitting the analyses

This folder holds the code, configuration, protocols and aggregate run records that produced the
analyses in the article, as they were run. It does not hold any input data, embeddings,
predictions or fitted models. To refit, obtain the SoluProt distribution and the pinned pLM
checkpoints (see the top-level `README.md`) and work through the stages below in order.

[`../docs/STAGE_RECORD.md`](../docs/STAGE_RECORD.md) states what each stage executed and how it is tied to the
reported numbers. [`../docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md) gives software versions and seeds.

## Where each item in the article's Code availability statement is

| Item | Code | Configuration or record |
|---|---|---|
| Entity grouping | MMseqs2 command in `01_data_preparation/src/assign_solubility_splits.py` | `01_data_preparation/records/soluprot_partition_manifest.json` |
| Fold and partition assignment, seeds | `01_data_preparation/src/assign_solubility_splits.py` (outer partition); `02_nested_fitting/analysis_addon/src/split_contracts.py`, `validate_v4_splits.py` (inner and sensitivity folds) | `02_nested_fitting/run_20260810_v4/manifests/split_validation.json` |
| B0 and B1 features | `01_data_preparation/src/build_solubility_features.py` (B0); `02_nested_fitting/analysis_addon/src/sequence_features.py` (B1, 400 ordered dipeptides) | `01_data_preparation/records/solubility_transparent_features.json` |
| Embedding extraction, pinned checkpoints, pooling and truncation | `01_data_preparation/src/embed_solubility_plm.py` (ProtT5, ESM-2 650M); `03_distillation/a2_rescue_frozen_esm2_35m/src/evaluate_correct_frozen_esm_ensemble.py` (ESM-2 35M) | `01_data_preparation/records/embeddings/`; `03_distillation/f_small_plm_esm2_35m/runs/ENCODER_IMPORT_QA.json` |
| Nested fitting and hyperparameter grids | `02_nested_fitting/analysis_addon/src/` (`run_soluprot_reanalysis.py`, `nested_models.py`, `resumable_models.py`) and the runtime helpers in `02_nested_fitting/run_20260810_v4/private_tools/` | `02_nested_fitting/analysis_addon/config/analysis_config_v4.yaml`; frozen protocol `analysis_addon/protocol/RETROSPECTIVE_ANALYSIS_FREEZE_20260810_V4.md` |
| Cross-fitted teacher targets | `03_distillation/a_fixed_audit_tripeplite/src/teacher_crossfit.py`; `03_distillation/c_teacher_increment_linear/src/increment.py` | each folder's `protocol/` |
| Distilled student; independently tuned and matched hard-only controls | `03_distillation/f_small_plm_esm2_35m/src/student_head.py`; linear, nonlinear and neural students in `c_…`, `d_…`, `e_…` | `runs/*/selection.json`, `runs/*/result.json` |
| Retrospective external evaluation | `04_external_evaluation/nesg/src/run_external.py`; exploratory arms in `nesg/exploratory_improvement/` | `nesg/protocol/PRE_EXECUTION_FREEZE.json`, `nesg/REPRODUCE.md` |
| Bootstrap evaluation | `02_nested_fitting/analysis_addon/src/clustered_stats.py`; independent re-implementation `04_external_evaluation/nesg/src/verify_results.py`; `03_distillation/f_small_plm_esm2_35m/src/increment.py`, `retention_uncertainty.py`; `05_screening_budget/screening_budget.py` | result files of each stage |
| Software versions and seeds | — | `../docs/ENVIRONMENT.md` |
| Stage-by-stage record | — | `../docs/STAGE_RECORD.md` |

## Order of the stages

1. **`01_data_preparation/`** — place the downloaded SoluProt and NESG files under
   `${WORKSPACE}/data/external/soluprot/`, check them against
   `records/soluprot_provenance_audit.json`, then run `prepare_solubility_benchmark.py`, MMseqs2
   with the recorded parameters, `assign_solubility_splits.py`, `build_solubility_features.py` and
   `embed_solubility_plm.py`. Default output locations are under `${WORKSPACE}/data/solubility/`.
2. **`02_nested_fitting/`** — the canonical B0, B1 and B1+ProtT5 comparison. Commands are in
   `analysis_addon/README.md`; use `config/analysis_config_v4.yaml`, which pins the input hashes the
   run checks before fitting.
3. **`03_distillation/`** — `a_…` and `b_…` are the two historical, rejected audits; `c_…` builds the
   cross-fitted teacher targets that `d_…`, `e_…` and `f_…` import. Each folder has its
   specification under `protocol/`.
4. **`04_external_evaluation/nesg/`** — see `REPRODUCE.md`.
5. **`05_screening_budget/`** — see `REPRODUCE.md`.

## Reading the code as run

- **Documents are records of their time.** Protocols, READMEs and reports are published as they were
  written during the analysis. Statements about what was "authorized", about manuscript
  integration or about results "not yet" available describe the internal workflow at that date; the
  article and `../docs/STAGE_RECORD.md` give the final status. Internal planning notes were not published.
- **Paths.** Nothing refers to a folder on the authors' computer. Local paths are written as
  `${WORKSPACE}` (a folder of your choice that holds `data/` and the run outputs) or `${HOME}`.
  Configuration files have a `paths:` block for you to fill in; `03_distillation/b_reconstruction`
  reads its root from the `STAM_WORKSPACE` environment variable; the small-pLM scripts read the
  ESM-2 35M checkpoint from your Hugging Face cache. Some stage scripts locate sibling stages and
  their run outputs (which are not published) by the folder names used during the analysis, for
  example `paper/peptide_solubility_review/analysis_addon/src`; point these at the corresponding
  folders of this repository and at your own run outputs. `../tools/verify_manifest.py` checks every
  published file against the manifest.
- **Run-time hash checks.** Preflight steps compare input files with the SHA-256 values pinned in
  the configuration. Inputs you rebuild from the original data should match where the step is
  deterministic; embeddings rebuilt on other hardware may not be bitwise identical, in which case
  the check must be updated deliberately rather than bypassed.
- **`SHA256SUMS.txt` files** also list files that are not redistributed here (embeddings, caches,
  fitted models), so `shasum -c` reports them as missing.
- **`vendor/` folders** contain copies of the authors' own modules taken into each stage so that it
  ran against a fixed version; they are not third-party code. `vendor/descriptors.py` is the B0
  descriptor function set.
- **`private_tools/`** is the name the runtime-helper folder had during the run; the helpers are
  published here.
