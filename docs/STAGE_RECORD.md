# Stage-by-stage execution record

This record states, for each step of the analyses in the article, whether it was executed end to
end on the full data, whether it was exercised only on reduced or synthetic inputs, and what
evidence ties the published code to the reported result. Where a record is missing, it says so.

**Terms.** *Full data*: the SoluProt development set (11,436 records; 10,715 sequence-identity
clusters) or the NESG test set (3,100 records). *Bound*: a SHA-256 recorded at run time matches the
as-run file; the authors checked every such match before release. Some published copies differ from
the as-run file because local paths are written as `${WORKSPACE}` or `${HOME}`, or because of a
declared edit; `tools/MANIFEST_SHA256.tsv` gives both hashes and the reason. Hash lists also cover
files that are not published here (input data, caches, fitted models, internal notes). *Unbound*: the code exists and is consistent with its output, but no
run-time hash or log ties this exact version to it.

Paths are relative to the repository root. File hashes for every published file are in
`tools/MANIFEST_SHA256.tsv`.

## 1. Data preparation (`level3/01_data_preparation/`)

| Step | Code | Executed on | Evidence | Gaps |
|---|---|---|---|---|
| 1a. Source files and checksums | `src/prepare_solubility_benchmark.py` | Full data, 4 Aug 2026 | `records/soluprot_provenance_audit.json` records the SHA-256 of every source file and the software versions. | No run log. The script's modification time (15:06) is later than its outputs (14:58), so the as-run version is unbound. |
| 1b. Sequence-identity grouping | MMseqs2 17-b804f; the command is recorded in `src/assign_solubility_splits.py` and `records/soluprot_partition_manifest.json` (`--min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode 1 -s 7.5`) | Full data | The resulting cluster table's SHA-256 and counts (10,715 clusters; 10,031 singletons; largest cluster 3) are in the partition manifest. | The MMseqs2 invocation itself was not logged. The parameters for clustering the external NESG set were not recorded; only its 30% file label survives. |
| 1c. Partition assignment and seeds | `src/assign_solubility_splits.py` (StratifiedGroupKFold, 7 folds, seed 20260804) | Full data | The partition file's SHA-256 (`28a63f9d…`) is in the partition manifest and is pinned in every downstream preflight record. | Code unbound (no code hash, no log); modification time precedes the output. |
| 1d. B0 features | `src/build_solubility_features.py` (Biopython ProtParam) | Full data | Output hashes and versions in `records/solubility_transparent_features.json`. The external-evaluation input audit recomputed these features independently and agreed to 1.4 × 10⁻¹⁴. | Code unbound (no code hash, no log); modification time precedes the outputs. |
| 1e. ProtT5 embeddings | `src/embed_solubility_plm.py`; `Rostlab/prot_t5_xl_half_uniref50-enc` at revision `94a6abc0…` | Full data (14,536 records, CPU) | `records/embeddings/soluprot_prott5_metadata.json`: checkpoint, revision, 900-residue chunks, residue-only mean pooling, output SHA-256 pinned downstream. | The script was edited at 18:31 on 4 Aug, after the ProtT5 output was written (17:46). The as-run version is unbound; the published version is the later one. |
| 1f. ESM-2 650M embeddings | Same script; `facebook/esm2_t33_650M_UR50D` at revision `08e4846e…` | Full data (13 Aug) | `records/embeddings/soluprot_esm2_metadata.json`. | Run after the edit above; no separate code hash. |
| 1g. ESM-2 35M embeddings (small-pLM student) | `level3/03_distillation/a2_rescue_frozen_esm2_35m/src/evaluate_correct_frozen_esm_ensemble.py` | Full development folds, on Apple MPS | Cache hash verified by `RESCUE_FINAL_SHA256_20260813.txt`. | Loaded **without a pinned revision**. Revision `6fbf070e…` and the checkpoint file hashes were pinned afterwards by four CPU re-extraction probes agreeing within 1.91 × 10⁻⁶ (`level3/03_distillation/f_small_plm_esm2_35m/runs/ENCODER_IMPORT_QA.json`). |

## 2. Nested fitting, the canonical ProtT5 result (`level3/02_nested_fitting/`)

| Step | Executed on | Evidence |
|---|---|---|
| B1 features (400 ordered dipeptides), nested logistic models, fold-local supervised PLS of ProtT5, hyperparameter grids (`analysis_addon/config/analysis_config_v4.yaml`) | **Full data, end to end**: 11,436 records, 7 outer × 5 inner folds | **Bound.** The source-tree hash of `analysis_addon/` (`37a743ee…`; `.py`, `.yaml` and `.json` files) equals the `code_hash` in `run_20260810_v4/manifests/preflight_manifest.json`; its published configuration files carry path placeholders, so the match holds for the as-run files. The five runtime helpers in `run_20260810_v4/private_tools/` match the hashes in `manifests/runtime_*_decision.json`. |
| Group bootstrap (10,000 replicates), 5 stability seeds, 50 label permutations | Full data | Result `run_20260810_v4/raw_results/internal_primary.json` (B0 0.704, B1 0.715, B1+ProtT5 0.787; gain +0.072, 95% CI [0.065, 0.079]). Run logs in `run_20260810_v4/logs/`, including a resume after a power interruption with checkpoint hashes checked before and after reload. |
| Release of the aggregate behind Figure 5(a) | — | `level1/scripts/release_internal_auc_s1.py`; the SHA-256 of the as-run script is recorded inside `level1/analysis_addon/results/aggregate_internal_auc_s1.json`. The published copy has one declared edit (a local folder name removed from its list of forbidden output strings), so it no longer matches that value. |

Figure 5(b), the exploratory gradient-boosted-tree ablation, is reproduced at Level 1 from
`level1/analysis_addon/results/exploratory_histgbm_plm_ablation_20260805.json`. Its upstream fitting
scripts used ESM-2 35M and ProtT5 without a pinned revision and left no run log; they are not part
of this release.

## 3. Distillation (`level3/03_distillation/`)

| Folder | What it is | Executed on | Evidence |
|---|---|---|---|
| `a_fixed_audit_tripeplite/` | First (historical, rejected) compact-student audit: cross-fitted teacher targets (`src/teacher_crossfit.py`) and the TriPepLite student | Development data, end to end (design on folds 2–6, gate on fold 0, confirmation on fold 1) | Aggregate `level1/analysis_addon/results/distillation_feasibility_s2.json`; its AUCs match the run's confirmation aggregate. The code was bound by a plan-time SHA-256 list and the aggregates by an execution SHA-256 list; those lists and the run logs are kept with the private run records. `dry_run/` holds **reduced-input** dry runs. No script writes the aggregate JSON. |
| `a2_rescue_frozen_esm2_35m/` | Frozen-encoder probes after the first audit; also writes the ESM-2 35M cache (step 1g) | Reduced ("smoke") and full-development runs, as named in each aggregate file | Aggregates and `RESCUE_FINAL_SHA256_20260813.txt`. |
| `b_reconstruction/` | Post hoc reconstruction of the second historical audit | Development data, end to end (3,065.5 s) | `results/run_20260822.log`; result byte-identical to `level1/analysis_addon/results/distillation_audit_reconstruction_20260822.json`. |
| `c_teacher_increment_linear/` | Cross-fitted teacher targets for the follow-up (`src/increment.py`), linear students, B1/context control | Folds 2–6, end to end | `runs/*/preflight.json` record source and input hashes, all matching the as-run files; `SHA256SUMS.txt`; `runs/*/result.json`. The 110 teacher-target files were hash-verified on import by the other folders. No run logs. |
| `d_nonlinear/`, `e_neural/` | Nonlinear and neural students against the same control | Folds 2–6, end to end | `runs/nested/preflight.json` (source hashes matching), `runs/nested/result.json`, `SHA256SUMS.txt`. No run logs. |
| `f_small_plm_esm2_35m/` | Small-pLM student (frozen ESM-2 35M encoder) with distilled, independently tuned hard-only and matched hard-only heads; engineering measurements | Folds 2–6 (8,198 rows; 7,653 groups), end to end. Engineering timing used a **synthetic** 500-residue input plus four raw-sequence probes. | `runs/nested/preflight.json` (12 source hashes matching), `SHA256SUMS.txt`, `runs/nested/result.json`, `runs/engineering/result.json`, `runs/retention_uncertainty.json`. No run logs. |

Figure 6 is drawn from `level1/analysis_addon/results/distillation_capacity_20260909.json`. The SHA-256
values of its five source results (`c_…/runs/final_scale`, `d_…/runs/nested`, `e_…/runs/nested`,
`f_…/runs/nested`, `f_…/runs/engineering`) are recorded in that file and match. **The script that
combined them was not kept.**

## 4. Retrospective external evaluation (`level3/04_external_evaluation/nesg/`)

| Step | Executed on | Evidence |
|---|---|---|
| Prespecified B1 vs B1+ProtT5 on NESG (Table S7, block 1) | **Full data, end to end**: 3,100 external records; fit 8,198 / selection 1,621 / calibration 1,617 | **Bound.** `protocol/PRE_EXECUTION_FREEZE.json` (22 file hashes, all matching), `run.log`, `results/external_retrospective.json` (B1 0.537, B1+ProtT5 0.540), `results/INDEPENDENT_VERIFICATION.json` by a separate implementation (`src/verify_results.py`). |
| Exploratory arms (Table S7, blocks 2–3) | Full data | `exploratory_improvement/FREEZE.json` (23 own and 43 parent hashes matching), `run.log`, `verification.log`. |
| Seal and contract tests | **Synthetic** inputs | `protocol/test_external_seal.log`, `exploratory_improvement/test_contracts.log`. |

## 5. Screening-budget re-expression (`level3/05_screening_budget/`)

Full data, end to end. `run.log`, `results.json`, and `SHA256SUMS.txt` (8 of 8 matching).

## 6. Level 1 figure and table regeneration

On 27 November 2026, every main figure, the graphical abstract and Supplementary Tables S3, S7 and S8
were regenerated from this repository alone and compared with the published artwork and tables: the
figures were pixel-identical and the tables byte-identical.

Values typed by hand rather than generated: Supplementary Tables S2, S4 and S6 (their values are in
`level1/analysis_addon/results/`), the numbers drawn in the graphical abstract, and the Method 15 thresholds
in Table S5, which come from the private PepSol Explorer build. No script writes
`pepsol_explorer_s2_demo.json` (made from the private application) or
`exploratory_histgbm_plm_ablation_20260805.json`.

## 7. Not executed

No pooled direct-aqueous-solubility predictor was fitted (the feasibility gate in
`level1/audit/FORMAL_GO_NO_GO_DECISION_20260804.md` was not met). No prospective or wet-laboratory
validation was performed. The educational CLI in `educational_cli/` was run only on the synthetic demo input.
