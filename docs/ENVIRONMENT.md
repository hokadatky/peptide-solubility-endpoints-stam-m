# Environment record

Every run record names the same platform, macOS 15.4 on Apple-silicon (arm64) hardware. The stages
used two Python environments. The version records below are taken from the run records published in this
repository; the file that each value comes from is named.

## Python

| Stage | Python | Key packages | Record |
|---|---|---|---|
| Data preparation, partitions, B0 features | 3.9.18 | numpy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, Biopython 1.85 | `level3/01_data_preparation/records/*.json` |
| pLM embeddings (ProtT5, ESM-2 650M) | 3.9.18 | torch 2.5.1, transformers 4.41.2, tokenizers 0.19.1, sentencepiece 0.1.99 | `level3/01_data_preparation/records/embeddings/*_metadata.json` |
| Nested fitting, bootstrap, external evaluation, screening budget | 3.10.17 (conda-forge) | numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.2.0 (pinned) | `level3/02_nested_fitting/run_20260810_v4/environment/sc8_environment.json`; `level3/04_external_evaluation/nesg/protocol/preflight.json`; `level3/05_screening_budget/results.json` |
| Distillation follow-up | 3.10.17 | numpy 1.26.4, scikit-learn 1.2.0, xgboost 3.1.3; torch 2.5.1 for the small-pLM encoder | `level3/03_distillation/*/runs/*/preflight.json`; `level3/03_distillation/e_neural/runs/nested/torch_environment.json`; `level3/03_distillation/f_small_plm_esm2_35m/runs/engineering/result.json` |
| Level 1 figure tables (S3, S7, S8) | ≥ 3.10 | standard library only | — |

`env/requirements.txt` covers the audit tests; `env/requirements-plm.txt` lists the packages for the
educational CLI and embedding extraction with minimum versions. Biopython (B0 features) and xgboost
(distillation baselines) are needed for Level 3 but are not in either file; install the versions above.

## R

Figures are drawn with R, ggplot2 and grid. Packages and versions are listed in
[`env/R_DEPENDENCIES.md`](../env/R_DEPENDENCIES.md).

## Protein language model checkpoints

| Hub identifier | Revision |
|---|---|
| `Rostlab/prot_t5_xl_half_uniref50-enc` | `94a6abc029ae13029317b140b7424e012bf8dfbf` |
| `facebook/esm2_t33_650M_UR50D` | `08e4846e537177426273712802403f7ba8261b6c` |
| `facebook/esm2_t12_35M_UR50D` | `6fbf070e65b0b7291e7bbcd451118c216cff79d8` (pinned after extraction; see `STAGE_RECORD.md`, step 1g) |

## Random seeds

| Use | Seed |
|---|---|
| Outer partition assignment | 20260804 |
| Global seed and group bootstrap (canonical analysis) | 20260809 |
| Stability repeats | 20260809–20260813 |
| Label permutation | 20261809 |
| TriPepLite fixed audit | 20260813 |
| Distillation follow-up bootstrap; external evaluation | 20260909 |
| External exploratory arms; screening budget | 20261016 |
| Synthetic PepSol demo record | 20260813 |

Each value is also stored in the configuration or result file of its stage.

## Generative AI tools

As declared in the article: OpenAI Deep Research, GPT-5.6 Sol Pro and OpenAI Codex with GPT-5.6 Sol
and GPT-5.6 Luna models (accessed 3–24 August 2026), and Anthropic Claude Code (accessed 5–24 August
2026; service-selected models); OpenAI Codex and Anthropic Claude Code were also used in
October 2026 during revision. The project's working records for this article name the Codex model
identifier `gpt-5.6-sol`; no record of the other identifiers was kept.
