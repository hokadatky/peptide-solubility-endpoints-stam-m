# PLM-to-specialist distillation plan — 2026-08-13

Status: **PROTOCOL AND ENGINEERING DRY RUN COMPLETE; SCIENTIFIC TRAINING NOT STARTED**.

## What problem this plan solves

The practical problem is not simply to maximize one retrospective AUC.  A
transparent sequence model is fast and distributable but misses some of the
context captured by protein language models (PLMs).  A large PLM improves the
internal SoluProt worked-example discrimination, but requiring a multi-gigabyte
encoder every time a collaborator submits a sequence makes routine use harder.

The proposed solution is a reusable deployment bridge:

```text
endpoint-labelled sequences
        |
        +--> lightweight transparent baseline
        |
        +--> large frozen PLM teacher(s), used during training only
                         |
             group-exclusive cross-fitted soft targets
                         |
                         v
             endpoint-specific lightweight student
                         |
          ~1 MB teacher-free NumPy model artifact
```

For this worked example, the endpoint is recombinant soluble-expression status
in *E. coli*.  It is not direct aqueous solubility of a purified peptide.

## Why this is knowledge distillation

The student is trained on both observed binary labels and the softened response
of a high-capacity PLM teacher.  The teacher is absent at deployment.  This is
task-specific **response knowledge distillation**.  In contrast, the existing
PLS-8 projection compresses an embedding but still requires ProtT5 at inference
and must not be called a distilled deployment model.

## Generality and claim boundary

The **scheme** is endpoint-portable: for a new peptide/property endpoint, define
the label and conditions, create homology- or chemistry-aware groups, cross-fit
an appropriate high-capacity teacher, and retrain a small endpoint-specific
student.  The same workflow can be adapted to expression, aggregation,
permeability, stability, toxicity, or other prediction tasks.

The **fitted weights are not endpoint-portable**.  No claim may state that the
SoluProt student predicts another property, or that transfer to another
endpoint has already been demonstrated.  Modified/non-canonical peptides also
need a representation that explicitly preserves their chemistry.

## Current evidence

- Frozen retrospective V4 internal result: order-aware B1 AUC 0.7152;
  B1 + fold-local ProtT5 PLS AUC 0.7872.
- Old ESM-2 hybrid PoCs are not publication-grade because they used ESM-2 35M,
  512-residue truncation, special-token-contaminated pooling, and row rather
  than cluster bootstrap.
- Correct ESM-2 650M extraction is incomplete: 777/1,456 shards, 4,539/14,536
  records touched; 679 shards remain.
- CPU/float32 recomputation of existing shards 0 and 776 reproduced their
  stored residue sums exactly in a bounded parity dry run; historical device
  provenance is nevertheless recorded as unavailable rather than inferred.
- TriPepLite-v1 engineering dry run: 265,633 trainable parameters, 988,428-byte
  ephemeral NumPy artifact, Torch/NumPy probability difference < 1e-6.
- There is currently **no scientific AUC for the distilled student**.

## Entry points

- Frozen contract: `config/DISTILLATION_CONFIG_20260813.json`
- Scientific freeze: `protocol/DISTILLATION_ANALYSIS_FREEZE_20260813.md`
- Endpoint-portable scheme: `protocol/ENDPOINT_PORTABLE_DISTILLATION_BLUEPRINT_20260813.md`
- Student implementation: `src/tripeplite.py`
- Leakage-safe teacher implementation: `src/teacher_crossfit.py`
- Bounded stage runner: `src/run_distillation_stage.py`
- Dry run: `src/run_dry_run.py`
- ESM resume parity audit: `src/audit_esm_existing_cpu_parity.py`
- Static handoff validator: `src/validate_plan.py`

The S0 package, S1 package, V4 analysis, current application, and manuscripts
are protected.  Nothing in this plan authorizes their modification.
