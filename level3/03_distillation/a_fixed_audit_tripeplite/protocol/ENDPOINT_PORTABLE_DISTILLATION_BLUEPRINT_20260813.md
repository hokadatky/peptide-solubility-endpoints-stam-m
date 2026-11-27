# Endpoint-portable PLM-to-specialist blueprint

Date: 2026-08-13  
Status: method blueprint; only the SoluProt soluble-expression worked example is
within the present validation plan

## 1. The need and the design response

| User need | Design response | Evidence required before release |
|---|---|---|
| A useful predictor should recover information missed by inexpensive classical features. | Fit a high-capacity PLM teacher with the transparent/order-aware block retained as the reference. | Teacher and reference evaluated on identical group-exclusive rows. |
| A collaborator should not install or run a multi-gigabyte PLM for every sequence. | Use the PLM only during training and distil its endpoint response into a small sequence model. | Same-architecture hard-only control, teacher-gain retention, and teacher-free artifact inspection. |
| One pasted sequence should be scoreable immediately on an ordinary CPU. | Export a NumPy specialist pack containing only student weights, descriptor scaling and an endpoint card. | Torch/NumPy parity, size, latency, determinism and privacy gates. |
| Different assays or practical questions should not be collapsed into one ambiguous 'solubility' number. | Register one independently trained specialist pack per precisely defined endpoint or condition. | Endpoint-specific data, grouping, validation and model card for every pack. |
| A promising retrospective number should not be produced by homolog leakage or repeated tuning. | Generate teacher targets by group-exclusive cross-fitting, use one design gate, then one untouched confirmation set. | Zero group overlap, immutable selection record and paired group bootstrap. |
| The method should transfer beyond the first worked example. | Keep the seven-step protocol and runtime interface; replace the endpoint-specific data, teacher head, loss and specialist weights. | New independent validation for each endpoint; no reuse of SoluProt performance claims. |

## 2. Backcast from the desired user experience

Desired inference interaction:

```text
collaborator supplies one canonical amino-acid sequence
                         |
             shared validation/feature engine
                         |
       +-----------------+------------------+
       |                 |                  |
endpoint pack A    endpoint pack B    endpoint pack C
score + meaning    score + meaning    score + meaning
domain warning     domain warning     domain warning
```

An **endpoint pack** is the unit of scientific validity.  It contains:

1. compact student weights;
2. training-derived descriptor scaling;
3. a machine-readable endpoint identifier;
4. assay/condition and population definitions;
5. training-domain and applicability limits;
6. validation metrics with uncertainty;
7. artifact hash and software-schema version.

The shared runtime may calculate sequence features once and execute several
packs.  The score labels must remain separate.  For example, recombinant
soluble-expression probability in *E. coli* must not be displayed as measured
aqueous solubility at a stated pH.  A multi-endpoint interface is therefore a
registry of validated specialists, not multiple names attached to one fitted
model.

### Current boundary

The present run can produce at most one candidate pack:

```text
recombinant_soluble_expression_binary_ecoli
```

It cannot scientifically populate direct aqueous solubility, aggregation,
permeability, stability or toxicity cards.  Those entries remain unavailable
until their own endpoint-specific datasets and confirmations exist.

## 3. Reusable seven-step training scheme

For a dataset with sequence (x_i), observed endpoint (y_i), measurement
conditions (c_i), and leakage group (g_i):

1. **Freeze the endpoint contract.** Define what was measured, units or class,
   assay conditions, population, duplicate/conflict policy and exclusions.
2. **Freeze the validation groups.** Choose homology-, scaffold-, donor-,
   batch- or chemistry-aware groups appropriate to the intended use.
3. **Measure the lightweight reference.** Evaluate a transparent or otherwise
   deployable baseline on group-exclusive splits.
4. **Construct a high-capacity teacher.** Use frozen PLM representations and a
   task head fitted only inside each training split; compare candidate teachers
   using one prespecified design gate.
5. **Create cross-fitted soft targets.** The teacher that scores row (i) must
   have seen neither row (i) nor any member of (g_i).
6. **Train and confirm one specialist.** Fit the frozen student with hard labels
   plus teacher responses and compare it with the identical hard-only student
   on an untouched confirmation set.
7. **Export and audit the endpoint pack.** Remove the teacher; check accuracy
   retention, calibration-relevant metrics, applicability domain, size,
   latency, numerical parity, privacy and artifact identity.

This sequence is the transferable contribution.  The fitted weights and the
observed SoluProt AUC are not transferable contributions.

## 4. What changes for another prediction task

| Component | Binary endpoint | Continuous endpoint | Multiclass/ordinal endpoint |
|---|---|---|---|
| Observed target | Precisely defined class and decision context | Units, censoring and assay range | Ordered or nominal class definition |
| Teacher output | Cross-fitted class probability/logit | Cross-fitted mean, quantiles or distribution parameters | Cross-fitted class distribution or cumulative logits |
| Hard loss | Binary cross-entropy or prespecified task loss | Prespecified regression/censoring loss | Cross-entropy or ordinal loss |
| Soft loss | Temperature-scaled response loss | Loss to teacher response/distribution | Distributional or ordinal response loss |
| Validation | AUC/AP plus log loss/Brier | MAE/RMSE plus calibration/coverage as appropriate | Discrimination plus proper scoring rule |
| Grouping | Endpoint-appropriate, often sequence cluster | Same principle | Same principle |

These alternatives are design interfaces, not implemented claims.  The current
code implements binary canonical-sequence classification only.

## 5. Two valid ways to represent experimental settings

### A. Separate specialist per setting — default and safest

Train distinct packs when, for example, pH, temperature, host or assay protocol
defines meaningfully different targets.  This is preferred when setting
coverage is sparse or unbalanced.  The UI can score all installed packs after
one sequence submission while keeping their meanings explicit.

### B. One condition-aware specialist — only with suitable data

Supply both sequence and standardized condition variables to a jointly trained
student when the dataset spans those conditions sufficiently and the intended
interpolation/extrapolation domain is prespecified.  Split design must prevent
sequence and condition leakage, and confirmation must test the intended
sequence-by-condition generalization.  The present SoluProt data do not
authorize this claim.

## 6. Minimum acceptance contract for every new endpoint pack

A pack may enter a distributable registry only when all of the following are
true:

- endpoint and condition wording is unambiguous;
- train/design/confirmation roles were frozen before confirmation;
- the relevant group-overlap checks are zero;
- hard-only and distilled students use the identical architecture and training
  schedule apart from the soft-target term;
- confirmation metrics and paired group-level uncertainty are frozen;
- teacher-gain retention is reported from its component values, not alone;
- model size, CPU latency, inference parity and privacy pass their gates;
- applicability limitations are shown beside the score;
- the pack has a unique version and SHA-256;
- scientific and deployment owners approve release.

If the teacher adds no endpoint value, there is no PLM gain to distil.  If the
distilled student does not beat the hard-only student, the endpoint may still
have a compact predictor, but successful knowledge transfer has not been
established.

## 7. Manuscript-ready central message

### Japanese

> 大規模PLMを常時配布するのではなく、学習時だけendpoint-specific teacherとして
> 利用し、相同clusterを除外したsoft targetを小型specialistへ蒸留する。利用者は配列を
> 一度入力するだけで、登録済みendpointごとに意味と適用限界が明示されたscoreをCPU上で
> 得られる。同じ工程は他の予測課題にも再利用できるが、重みと性能はendpointごとに再学習・
> 再検証する。

### English

> Large PLMs are used as endpoint teachers during training, not distributed as
> routine inference dependencies. Group-exclusive cross-fitted responses are
> distilled into compact specialists, each released with an explicit endpoint
> card. A shared runtime can score one submitted sequence with multiple
> independently validated specialist packs. The workflow is reusable across
> endpoints; the fitted weights and performance are not.

## 8. Claim boundary

Allowed now:

> We define a reusable, leakage-controlled route from a high-capacity PLM
> teacher to a compact endpoint-specific sequence scorer.

Allowed only after the present confirmation passes:

> In the retrospective SoluProt soluble-expression worked example, the compact
> student retained [VALUE] of the selected teacher's AUC gain and passed the
> prespecified deployment gates.

Not allowed without new endpoint data:

- 'one universal model predicts all forms of peptide solubility';
- 'the SoluProt weights transfer to aggregation, permeability or toxicity';
- 'multiple assay settings are already supported';
- 'direct aqueous solubility is predicted';
- 'cross-endpoint accuracy has been demonstrated'.
