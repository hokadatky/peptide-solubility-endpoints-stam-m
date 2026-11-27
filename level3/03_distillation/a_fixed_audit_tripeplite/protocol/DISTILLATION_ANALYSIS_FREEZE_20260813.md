# Frozen protocol: PLM-to-specialist response distillation

Date: 2026-08-13  
Run ID: `soluprot_distillation_20260813_v1`  
Status: **retrospective, data-informed internal extension; not a preregistration**

## 1. Scientific question

Within the previously defined SoluProt recombinant soluble-expression endpoint,
can the task-relevant prediction behaviour of a large frozen PLM teacher be
transferred to a small sequence-to-score model that no longer requires a PLM at
inference?

This question has three distinct parts:

1. **Representation value:** does a PLM teacher discriminate better than the
   lightweight transparent/order-aware reference?
2. **Distillation value:** does adding cross-fitted teacher soft targets improve
   a fixed small student over the same student trained only on hard labels?
3. **Deployment value:** how much of the teacher gain is retained after the PLM
   is removed, and what are the artifact-size and CPU-latency costs?

The estimand is not direct aqueous peptide solubility.  It is discrimination of
retrospective recombinant soluble-expression labels in *E. coli*.

## 2. Motivation and intended contribution

The V4 worked example showed a gap between a distributable order-aware baseline
and a PLM-assisted model.  B1 achieved ROC AUC 0.71521248597, whereas B1 plus
fold-local ProtT5 PLS achieved 0.78721442119 on the same seven-fold,
homology-controlled internal OOF rows.  The observed difference motivates use
of PLM information; it does not by itself make PLM inference convenient.

The contribution tested here is therefore a **custom task-response compression
scheme**:

- use the large PLM only to construct an accurate endpoint teacher;
- create soft targets without allowing each target row or its 30%-identity
  cluster into the teacher fit that predicts it;
- train a fixed endpoint-specific student on raw sequence, transparent
  descriptors, hard labels, and the cross-fitted soft targets;
- export only the student weights and train-derived descriptor scaling;
- evaluate the teacher, hard-only student, and distilled student on exactly the
  same unused rows.

This is response knowledge distillation, not quantization of PLM weights and not
PLS representation compression.

## 3. Evidence status at freeze

### 3.1 Established evidence

- Development rows: 11,436.
- 30%-identity groups: 10,715.
- Labels: 5,718 negative and 5,718 positive.
- Fixed outer folds: seven, with each group wholly contained in one fold.
- ProtT5 final embedding: 11,436 development rows × 1,024 dimensions, aligned.
- V4 B1 AUC: 0.71521248597.
- V4 B1 + ProtT5 PLS AUC: 0.78721442119.

### 3.2 Incomplete evidence

- ESM-2 650M extraction: 777 of 1,456 shards exist; 679 remain.
- No corrected ESM-2 or ProtT5+ESM-2 V4-matched teacher result exists.
- No scientifically evaluated distilled-student result exists.
- No student external-transfer result exists.

### 3.3 Invalid evidence for final claims

The old ESM-2/hybrid scripts are PoCs only.  They used ESM-2 35M, truncated
sequences at 512 residues, included tokenizer-added boundary tokens in pooling,
used an unmatched learner/representation comparison, and performed a row-level
bootstrap.  Their numerical results may motivate the present fixed experiment
but may not appear as final performance evidence.

## 4. Fixed data roles: backcast from the required answer

The required final answer is an unbiased comparison between hard-only and
distilled versions of the same student, after one bounded teacher choice.  The
existing seven outer folds are assigned fixed roles:

| Role | Folds | Permitted use |
|---|---:|---|
| Design training | 2, 3, 4, 5, 6 | Cross-fitted teacher targets and student fitting |
| Design gate | 0 | One predeclared choice: hybrid teacher or ProtT5 fallback |
| Confirmation | 1 | One final internal evaluation; no model choice |

For confirmation, the model is refit using folds 0 and 2–6; fold 1 remains
untouched until all choices are fixed.  Fold 1 may never affect architecture,
teacher selection, alpha, temperature, epoch count, seed choice, threshold, or
claim wording.

The historical names `model_selection` and `calibration` are data provenance
labels.  No probability recalibration is introduced in this extension because
fold 1 is repurposed as a strictly untouched confirmation fold.  Scores are
reported as raw model outputs, and calibration metrics are descriptive only.

## 5. Teacher definition

### 5.1 Shared downstream contract

Each teacher uses:

- B1: 25 transparent variables plus 400 ordered-dimer frequencies;
- frozen, residue-only arithmetic-mean PLM embeddings;
- special tokens excluded;
- fold-local standardization;
- fold-local supervised PLS with exactly 8 components;
- L2 logistic regression, `C=1000`, `lbfgs`, `max_iter=20000`;
- no hyperparameter search in this extension.

PLS-8 and C=1000 are fixed from the earlier V4 development exercise.  Fixing
the same downstream head for both teacher candidates isolates the added ESM-2
representation as far as this retrospective design permits.

### 5.2 Candidate teachers

1. `PROTT5_TEACHER`: B1 plus ProtT5-XL embedding.
2. `PROTT5_ESM2_TEACHER`: B1 plus the concatenated ProtT5-XL and ESM-2 650M
   embeddings, followed by one joint fold-local PLS-8 projection.

Checkpoint/revision contracts are in the JSON config.  The hybrid teacher is
not assumed to be better.

## 6. How soft targets are generated without target leakage

For any stage-training fold set \(F\), create one teacher prediction for every
training row as follows:

1. For each fold \(k \in F\), fit the entire teacher pipeline on rows whose fold
   lies in \(F \setminus \{k\}\).
2. Predict only rows in fold \(k\).
3. Verify that no `cluster_id` occurs in both the fit and target rows.
4. Concatenate the fold-specific predictions in original row order.
5. Clip probabilities to `[1e-6, 1-1e-6]` only for stable logit conversion.

Thus the hard label of row \(i\), and the hard labels of every sequence in its
30%-identity group, are absent from the teacher fit that creates the soft target
for row \(i\).  Fitting one teacher on all training rows and then predicting the
same rows is prohibited because its in-sample confidence can be memorized by the
student.  Reusing the old seven-fold OOF teacher scores is also prohibited for
confirmation training: those scores were produced under a different model
selection contract and do not implement the present fold-role freeze.

## 7. Fixed student: TriPepLite-v1

### 7.1 Input

- canonical amino-acid sequence;
- the same 25 deterministic B0 descriptors, calculated from the sequence;
- no PLM embedding at inference;
- no teacher probability at inference.

Non-canonical residues, modifications, cyclization and stereochemistry fail
closed.  Adapting the scheme to modified peptides requires a chemistry-aware
student representation and a new dataset.

### 7.2 Architecture

1. Encode every overlapping canonical tripeptide exactly as an integer from 0
   to 7,999; no hashing collisions.
2. Use one shared 8,001 × 32 `EmbeddingBag` table for:
   - the whole sequence,
   - the first 32 residues,
   - the last 32 residues.
3. Mean-pool each bag and apply a separate LayerNorm.
4. Standardize 25 B0 descriptors using training rows only; project 25 → 32 with
   a linear layer and SiLU.
5. Concatenate the three 32-dimensional sequence vectors and the 32-dimensional
   descriptor vector; apply LayerNorm.
6. Apply `128 → 64`, SiLU, dropout 0.10, and `64 → 1` logit.

Exact trainable parameter count: 265,633.  The architecture is deliberately
small, deterministic in shape, able to process the full observed length range,
and contains local order plus N/C-terminal context without a model zoo.

### 7.3 Training

- optimizer: AdamW;
- learning rate: 0.003;
- weight decay: 0.0001;
- betas: (0.9, 0.999);
- epsilon: 1e-8;
- gradient clipping: L2 norm 1.0;
- batch size: 128;
- epochs: exactly 40;
- CPU primary device; one Torch thread;
- primary seed: 20260813;
- sensitivity seeds: 20260814 and 20260815, confirmation only;
- no early stopping and no best-checkpoint selection;
- no class weights;
- no hyperparameter search.

The hard-only and distilled models have identical architecture, initialization
seed, optimizer, minibatch order, and epoch count.  Only the loss differs.

## 8. Knowledge-distillation objective

Let \(y_i\) be the observed binary label, \(z_i^S\) the student logit, and
\(p_i^T\) the cross-fitted teacher probability.  With temperature \(T=2\):

\[
q_i^T(T) = \sigma\left(\frac{\operatorname{logit}(p_i^T)}{T}\right),
\qquad
q_i^S(T) = \sigma\left(\frac{z_i^S}{T}\right).
\]

The hard loss is ordinary binary cross-entropy:

\[
L_{hard} = -\frac{1}{n}\sum_i
\left[y_i\log\sigma(z_i^S)+(1-y_i)\log(1-\sigma(z_i^S))\right].
\]

The soft loss is binary cross-entropy against the softened teacher response:

\[
L_{soft} = -\frac{1}{n}\sum_i
\left[q_i^T\log q_i^S+(1-q_i^T)\log(1-q_i^S)\right].
\]

The distilled objective is fixed to:

\[
L = (1-\alpha)L_{hard} + \alpha T^2 L_{soft},
\qquad \alpha=0.5,\ T=2.
\]

The \(T^2\) factor retains the usual gradient-scale correction.  The hard-only
control sets \(\alpha=0\).  Alpha and temperature may not be changed after any
fold-0 or fold-1 result is viewed.

## 9. Fixed arms

| Arm | Purpose |
|---|---|
| B1_ORDER_LOGIT | Lightweight reference |
| TRIPEPLITE_HARD | Same student, labels only |
| PROTT5_TEACHER | Large single-PLM teacher |
| PROTT5_DISTILLED_TRIPEPLITE | Response distilled from ProtT5 teacher |
| PROTT5_ESM2_TEACHER | Large hybrid teacher |
| PROTT5_ESM2_DISTILLED_TRIPEPLITE | Response distilled from hybrid teacher |

No TCN, transformer student, LoRA, attention pooler, alternate k-mer length,
alternate alpha/temperature, ensembling, or post-hoc architecture search is
allowed in this run.

## 10. One bounded hybrid-selection gate

Fold 0 is used once.  Hybrid is selected only if all four conditions hold:

- hybrid teacher log loss improves by at least 0.005 versus ProtT5 teacher;
- hybrid-distilled student log loss improves by at least 0.005 versus the
  ProtT5-distilled student;
- hybrid teacher AUC is not more than 0.005 below the ProtT5 teacher AUC;
- hybrid-distilled student AUC is not more than 0.005 below the
  ProtT5-distilled student AUC.

Otherwise the teacher is ProtT5.  These are engineering selection thresholds,
not statistical significance claims.  No bootstrap or p-value is used to pick
the teacher.  After the JSON decision is written it is immutable.

## 11. Confirmation estimands and metrics

On fold 1, the primary scientific contrast is:

```text
selected distilled TriPepLite − hard-only TriPepLite
```

Primary metric: ROC AUC.  Supporting metrics: average precision, log loss and
Brier score.  All contrasts use paired cluster bootstrap with 10,000 replicates
and seed 20260813.  For loss/Brier, positive gain means the candidate has lower
loss.

Secondary descriptive quantities:

- selected teacher versus B1;
- selected distilled student versus B1;
- student versus teacher performance gap;
- teacher-gain retention:

\[
R = \frac{AUC_{student}-AUC_{B1}}
         {AUC_{teacher}-AUC_{B1}}.
\]

If the denominator is non-positive, retention is undefined.  The predeclared
deployment target is \(R \ge 0.80\), but this target is not a null-hypothesis
test and must be reported with the actual component AUCs.

## 12. Deployment gates

A fitted student is deployment-eligible only if:

- confirmation outputs pass all privacy and finiteness checks;
- no PLM, teacher outputs, labels, sequences, IDs, or row-level features are in
  the artifact;
- NumPy and Torch probabilities agree within maximum absolute error 1e-6;
- compressed model artifact is no larger than 5 MiB;
- median end-to-end CPU latency for one 500-residue canonical sequence is no
  greater than 100 ms on the audited machine;
- the model card states the exact endpoint and exclusion of direct aqueous
  solubility;
- applicability-domain warnings from the current app are retained;
- the author explicitly approves replacing or augmenting the app model.

Failure of the 80% retention target does not invalidate the scientific result;
it prevents presenting the student as a successful high-fidelity deployment
surrogate.

## 13. Portability to other prediction tasks

The reusable unit is the protocol, not the fitted SoluProt weights.

### Components that remain the same

- large teacher used only during training;
- group-exclusive cross-fitted soft targets;
- identical hard-only student control;
- frozen alpha, temperature and architecture before confirmation;
- one selection set followed by one untouched confirmation set;
- performance-retention, size, latency and privacy audits;
- teacher-free export.

### Components that must be redefined for each endpoint

- biological/chemical endpoint and measurement conditions;
- binary, ordinal or continuous target definition;
- duplicate/conflict rules;
- grouping unit and similarity threshold;
- baseline features and application domain;
- appropriate teacher and its calibration;
- loss for the observed endpoint;
- validation cohort and practical margin.

For continuous outcomes, replace hard BCE with a prespecified regression loss
(for example Huber loss) and the soft BCE with loss to a cross-fitted teacher
response/distribution.  This extension is conceptually specified here but not
implemented or validated in the present binary-classification code.

Permissible wording: 'The scheme can be adapted to other endpoint-defined
peptide property predictors by retraining both teacher head and student.'

Prohibited wording: 'The SoluProt student is a universal peptide solubility or
developability model' or 'transfer to other endpoints was demonstrated.'

## 14. External evaluation

The historical external dataset has already influenced prior development.
Opening it again cannot create a locked prospective test.  External application
is optional, must occur only after the complete internal confirmation artifact
is frozen, and must be described as retrospective cross-database transfer.
No teacher choice, student retraining, threshold adjustment or claim repair is
allowed after viewing it.

## 15. Protected roots and privacy

The following are read-only in this protocol:

- `paper/peptide_solubility_review_submission_s0_20260812/`
- `paper/peptide_solubility_review_submission_s1_auc_author_review_20260812/`
- `paper/peptide_solubility_review/analysis_addon/`
- `paper/general_peptide_solubility_scoping/analysis_addon_run_20260810_v4/`
- `app/`

Private row-level outputs must remain under:

`paper/general_peptide_solubility_scoping/analysis_distillation_run_20260813_v1/`

Tracked plan outputs may contain only aggregate results, contracts, source code,
tests and documents.  No record ID, sequence, label, teacher target, fold-level
row table or prediction may be copied into them.

## 16. Stop rules

Stop immediately and report the exact phase if any of the following occurs:

- input SHA-256, row, label, group, fold or embedding-order mismatch;
- any group overlap between a teacher fit and its soft-target rows;
- ESM-2 final validator failure;
- non-finite loss, gradient, weight or prediction;
- existing checkpoint has a different frozen contract hash;
- a command would overwrite a prior result rather than resume it;
- fold 1 is read for model selection;
- any protected root changes;
- any row-level material appears outside the private run root;
- code requires an unplanned model/parameter search;
- the endpoint is described as direct aqueous solubility.

Scientific underperformance is not a software failure.  Record it honestly and
apply the claim ceiling; do not start a rescue search.

## 17. Manuscript claim ceiling

Before scientific execution, only the method and motivation may be described.
After confirmation:

- if distillation improves the same student and deployment gates pass, describe
  an endpoint-specific lightweight surrogate and report exact retention;
- if the student is compact but does not improve over hard-only training,
  describe a tested but unsuccessful distillation attempt;
- if the student improves but retains little teacher gain, call it a speed/
  accuracy trade-off, not PLM-equivalent performance;
- never elevate this one retrospective endpoint to 'world-best aqueous peptide
  solubility prediction'.

