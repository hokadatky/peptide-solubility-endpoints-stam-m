# Formal GO/NO-GO decision: general-peptide solubility and protein language models

**Decision date:** 4 August 2026
**Scope:** chemically defined peptides, direct experimentally measured aqueous
solubility, and protein-language-model (pLM) methods
**Decision:** **NO-GO for a general predictive benchmark; GO for an
endpoint-centred Introductory Review or Focus Issue Review with a reproducible
public-data audit**

## Executive decision

The proposed claim that a frozen pLM improves the prediction of aqueous
solubility for general peptides cannot presently be tested credibly with the
located public data. The barrier is not a lack of model capacity. It is that
`solubility` denotes several non-equivalent observations and that public direct
peptide measurements are small, scaffold-concentrated, chemically modified,
condition-specific, or censored.

The scientifically stronger article is therefore:

> **What Do Protein Language Models Predict About Peptide Solubility?**

It should explain how endpoint definition determines the meaning of a pLM
prediction, audit what published `solubility` datasets actually measure, and map
the cases that are easy, context-limited, condition-limited, phase-boundary
limited, or outside a canonical pLM's chemical vocabulary.

This is not a retreat from the requested depth. The central materials-method
contribution becomes a distinction among label, representation and physical
phenomenon. A higher benchmark score is useful only after those three have been
made commensurate.

## What `solubility` can mean

| Name | Operational meaning | Dominant determinants | Legitimate prediction claim |
|---|---|---|---|
| Intrinsic or equilibrium aqueous solubility | Equilibrium concentration of a defined chemical species under stated pH, buffer, ionic strength, temperature and solid/phase conditions | Ionization, hydration, solid-state chemical potential, molecular association and phase behaviour | Condition-specific thermodynamic solubility |
| Apparent or kinetic solubility | Concentration remaining after a stated preparation, incubation and separation protocol | Dissolution and precipitation kinetics, stock solvent, supersaturation, aggregation, filtration and assay time | Protocol-specific apparent concentration |
| Soluble fraction after recombinant expression | Fraction recovered in a supernatant after expression, lysis and centrifugation | Expression, folding, degradation, host, construct, tags, process conditions and recovery | Recombinant-production soluble recovery |
| Production-pipeline status | Database stage such as `soluble`, `purified` or a later stage | Biology, laboratory workflow, missingness and pipeline decisions | Operational production-stage classification |
| Aggregation/processability/phase proxy | Turbidity, aggregation propensity, precipitation onset, flow, gelation, LLPS or self-assembly | Concentration, time, formulation, surfaces, kinetic pathway and phase transitions | The named proxy only |

These labels are scientifically related but are not interchangeable. In
particular, PROSO II, PeptideBERT, SoluProt, NetSolP, eSOL and PLM_Sol do not
provide direct measurements of general-peptide equilibrium aqueous solubility.
They remain useful in the review as examples of endpoint-specific pLM use.

## Verified public-data audit

| Source | Direct records verified | Chemistry and endpoint | Why it cannot support the proposed general benchmark |
|---|---:|---|---|
| CamSol-PTM | 34 experimental outcomes in the Figure 3 source-data table; 37 strings in a separate supplementary sequence table, without a row key establishing a one-to-one join | Three related scaffolds; two precipitant-midpoint assays and one ultracentrifugation-supernatant assay; most variants contain non-natural residues | The numerical outcomes cannot be pooled as one `logS` target; only one sequence string is a canonical one-letter sequence |
| Stapled-peptide design-rules dataset | 234 exact numerical values among 418 analogue rows | Kinetic solubility after DMSO-to-PBS dilution, filtration and UPLC/DAD; close stapled and modified analogue series | Canonical pLM inputs omit staples, D-residues and terminal chemistry; 139 values are missing and 45 are recorded as the categorical outcome `Poorly Behaved` |
| PEGASUS | 40 direct values | Cyclic peptides in HBSS, pH 7.4, after 18 h by LC-MS/MS; mostly N-methylated and deliberately enriched for high solubility | No canonical one-letter entries; one five-structure collision group loses four chemical distinctions when reduced to a bare sequence string; the massively parallel measurements are permeability proxies, not solubility |
| Claudin-5 study training panel | 16 qualitative labels | Eight `soluble` and eight `insoluble` peptides selected from the Antimicrobial Peptide Database and then subjected to molecular dynamics | The supplementary table supplies no direct row-level solubility measurement, assay condition or primary citation supporting those labels |

The three quantitative public sources contain at least 308 direct numerical
outcomes in aggregate, but that number is not a model-ready sample size. The
records describe different chemical objects and different measurement
operators. Pooling them would manufacture a target that no experiment measured.

The audit files preserve source checksums, repository commit where applicable,
row counts, missing/textual-value counts and representation limitations. No raw
peptide rows are reproduced in the narrative report.

## Literature-search boundary

A reproducible discovery search on 4 August 2026 returned:

- 272 PubMed records for peptide titles/abstracts mentioning aqueous,
  intrinsic or kinetic solubility;
- 302 Europe PMC records using the corresponding title/abstract query;
- three PubMed records for the deliberately narrow peptide–solubility–pLM
  title/abstract query; and
- 8,057 OpenAlex full-text search hits, of which the first 200 demonstrated that
  the query was too nonspecific for an eligibility count.

These are database-specific discovery counts, not deduplicated PRISMA counts.
The broad direct-solubility sets include formulations, conjugates, prodrugs,
solubilizing tags and individual medicinal-chemistry series. The narrow pLM
records include PeptideBERT and a 2026 gated-query-pooling study that reuse
production-stage labels; they do not establish a public general-peptide aqueous-
solubility benchmark.

## The deeper scientific story

### 1. Classically apparent cases

Charge state at the assay pH, hydrophobic burden, aromatic content, size and the
absence of a concentrated hydrophobic segment point in the same direction. A
transparent physicochemical model may be sufficient. A pLM should be required
to demonstrate a practically important incremental benefit rather than merely
produce a plausible score.

### 2. Cue-conflicted cases

Strong factors oppose one another: a highly charged peptide also has an aromatic
patch, or a globally polar sequence has a locally self-associating motif. These
are plausible targets for contextual sequence representations, provided the
training labels and chemistry are relevant.

### 3. Sequence-, conformation- and solid-state-context cases

Residue order, amphiphilic patterning, cyclization, stereochemistry,
N-methylation and terminal chemistry can change intramolecular hydrogen bonding,
solvent exposure, aggregation and crystal packing without a large change in
composition. Canonical ProtT5 and ESM-2 can represent residue order for canonical
linear sequences, but they cannot faithfully encode several of these chemical
distinctions.

### 4. Condition-sensitive cases

pH, buffer, ionic strength, temperature, concentration, incubation, stock
solvent, solid form and separation method can change the observed value. A
sequence-only pLM cannot infer an unrecorded condition. The appropriate advance
is condition-aware data and modelling, not necessarily a larger encoder.

### 5. Phase-boundary cases

Gelation, colloidal aggregation, precipitation, LLPS and ordered self-assembly
can make the measured supernatant concentration and apparent flow behaviour
protocol-dependent. Here `soluble` is not a stable material attribute but a
statement about a phase diagram and an observation protocol.

### 6. Unsupported chemistry

Non-natural residues, staples, D-residues, cyclic topology, conjugates and salt
forms may lie outside the pLM vocabulary. A model failure is then an input-
representation failure before it is a failure to learn solubility physics.

## Recommended article architecture

**Article type:** invited Focus Issue Review or Introductory Review. If the editor
prefers a Research Article, the original contribution must be described as a
reproducible endpoint-and-data audit, not as a pLM superiority benchmark.

1. **What is peptide solubility?** Define the measurement families and why they
   cannot be merged.
2. **From physicochemical rules to pLMs.** Review composition, descriptors,
   task-trained models, frozen encoders, fine-tuning and multimodal directions.
3. **What has actually been predicted?** Tabulate dataset provenance, endpoint,
   chemistry, model role, split design and permitted claim.
4. **A reproducibility audit of public direct-solubility data.** Present the four
   verified audits above and the benchmark NO-GO boundary.
5. **What makes prediction difficult?** Use the six-regime framework, with
   mechanistic examples involving local patterning, packing, conformation,
   conditions and phase behaviour.
6. **Where pLM evolution can help.** Distinguish canonical sequence context from
   chemical representation, condition awareness, structure/surface information,
   uncertainty and applicability-domain control.
7. **Minimum reporting standard.** Require endpoint name, units, conditions,
   chemical identity, censoring, duplicate policy, scaffold-held-out evaluation,
   calibration and compute cost.

## Display-item plan

1. **Figure 1:** endpoint map from chemical species and protocol to observed
   `solubility` label.
2. **Table 1:** evolution from physicochemical models to current pLM practice.
3. **Table 2:** prior pLM solubility studies, with the endpoint actually
   predicted and model role.
4. **Figure 2:** public-data audit showing size, chemistry support and assay
   comparability without pooling non-equivalent values.
5. **Figure 3:** difficulty map from classically apparent to context-, condition-
   and phase-limited cases.
6. **Table 3:** minimum reporting and benchmark standard.
7. **Figure 4:** future method stack: chemistry-aware identity plus sequence,
   conditions, phase state, uncertainty and experimental feedback.

No graphical abstract should be created until this endpoint-centred story is
accepted. The former recombinant-*E. coli* graphical abstract is scientifically
incompatible with this article.

## Claim boundary

Permitted:

> Published pLM studies show that learned sequence representations can improve
> selected production-related or proxy endpoints. Public evidence is not yet
> sufficient to establish a general improvement in directly measured peptide
> aqueous-solubility prediction.

Not permitted:

- `pLMs improve general-peptide aqueous-solubility prediction`;
- `PeptideBERT predicts intrinsic peptide solubility`;
- `308 public measurements form one harmonized benchmark`;
- `canonical residue embeddings encode cyclization, D-stereochemistry or
  N-methylation`;
- `aggregation, LLPS or self-assembly is equivalent to insolubility`.

## Conditions that would reopen the predictive benchmark

The benchmark may be reconsidered if an independently verified source provides
at least roughly 500 diverse, chemically identified peptides measured under a
commensurate aqueous-solubility protocol, or 150--499 records across several
independent scaffolds with sufficient metadata for scaffold-held-out analysis.
Chemical modifications must be represented explicitly; close analogues must be
kept within one split; and the outcome must remain condition-specific.
