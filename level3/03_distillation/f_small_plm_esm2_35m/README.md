# Frozen ESM35 head distillation — exploratory, 2026-09-09

Result: pooled outer OOF AUC 0.759675 vs independently tuned same-encoder hard-only 0.751377, B1 0.711589 and ProtT5 teacher 0.779931. Paired KD minus hard-only +0.008298, conditional 95% cluster-bootstrap CI [0.002368, 0.014181]. Teacher gain retention 70.36%, not all attributable to KD: hard-only already retains 58.22%. No new independent confirmation; development folds 2–6 were reused across method cycles. No promotion. Original 80% retention, 5 MiB and 1 ms gates remain unmet.

`protocol/SPEC.md` records fitting rules; `runs/nested/result.json` records accuracy; `runs/engineering/result.json` records complete raw-sequence timing; `runs/retention_uncertainty.json` records paired ratio uncertainty. Import and encoder QA manifests record provenance. Never substitute another project's targets, predictions or cache.

The student is frozen ESM-2 35M + B1 sequence features + distilled logistic head. Only the head receives teacher targets; this is not full encoder distillation. 33,762,666 predictive parameters, FP32 encoder/head files 135,092,558 bytes, CPU batch-1 500-aa median 242.966 ms. Runtime still requires its own language model, torch and transformers. FP16 export is storage-only and unvalidated.

## Local prediction prototype

From this directory, in the recorded prot5 environment (torch 2.5.1, transformers 4.41.2), with NumPy and safetensors installed:

```python
import sys
sys.path.insert(0, 'src')
from runtime import Student
s = Student('runs/encoder_export', 'runs/encoder_export', 'runs/head_export/fold6.npz')
score = s.predict(['ACDEFGHIKLMNPQRSTVWY'])
```

The tokenizer files and FP32 encoder are retained locally. Fold 6 is a prespecified representative development head, not a final all-data deployment fit. Inputs must contain canonical amino acids only. Above 1022 residues the encoder uses first 511 + last 511, while B1 describes the full sequence. The artificial junction and missing middle limit interpretation. This prototype is not a calibrated solubility assay or an OOD acceptance rule.

`src/student_head.py` reproduces fitting using retained same-project development features and teacher-target cache; `src/export_heads.py`, `src/export_encoder.py`, `src/benchmark.py`, and `src/retention_uncertainty.py` document derivation. Read the scripts' CLI and protocol before rerunning: nested fitting and encoder export overwrite their selected output paths if reused. Prefer a separate new run directory. No automatic refitting is necessary to review the saved result.

See `../peptide_solubility_review_submission_s20_floats_after_refs_20260827/DISTILLATION_METHODS_ADDENDUM_20260909.md` for the complete Japanese methods and limitations. Private per-row predictions and model weights are retained locally, not included in the manuscript-source ZIP.
