# Preliminary feasibility audit: measured aqueous solubility of general peptides

**Date:** 4 August 2026
**Status:** completed preliminary feasibility audit; not a systematic review and
not a manuscript
**Active scope:** chemically defined general peptides with direct experimental
aqueous-solubility measurements

## Interim decision

No single public dataset identified in the preliminary search supports a strong
general-peptide pLM benchmark. After file-level inspection of the most relevant
public sources, the decision is **NO-GO for predictive modelling and GO for an
endpoint-centred review with a reproducible public-data audit**. See
`FORMAL_GO_NO_GO_DECISION_20260804.md`.

The available candidates are scientifically useful but fragmented. They differ
in peptide class, chemical modifications, assay definition, solvent conditions
and scale. The largest advertised data streams in recent cyclic-peptide AI work
primarily measure hydrophobicity or permeability proxies; the directly measured
aqueous-solubility subsets appear much smaller. No recombinant-expression
dataset will be substituted.

## Candidate evidence discovered

| ID | Source | Direct endpoint | Preliminary scale/scope | Potential value | Blocking issue before inclusion |
|---|---|---|---|---|---|
| E001 | Oeller et al., CamSol-PTM, *Nature Communications* (2023), DOI `10.1038/s41467-023-42940-w` | Intrinsic/relative peptide solubility assessed experimentally at room temperature using scaffold-specific precipitation or separation protocols | 34 experimental outcomes in the Figure 3 source-data table; a separate supplementary table contains 37 sequence strings, without a row key establishing a one-to-one join | Best-documented public peptide-specific source; explicit sequences and source data | Three related scaffolds and three assay/scaffold targets cannot be pooled; most variants contain non-natural residues and only one sequence string is canonical one-letter input |
| E002 | Baker et al., PEGASUS, *Journal of Medicinal Chemistry* (2026), DOI `10.1021/acs.jmedchem.5c01836` | Aqueous solubility of selected cyclic peptides in HBSS at pH 7.4 after an 18-h incubation, quantified by LC-MS/MS | 40 direct values; 35 above 100 micromolar; 40 unique isomeric structures but only 36 bare sequence strings | Modern cyclic-peptide developability data with explicit chemistry and standardized assay description | No canonical one-letter entries; one five-structure collision group loses four chemical distinctions when reduced to a bare sequence string; selected high-solubility enrichment; the massively parallel data are permeability proxies |
| E003 | Zhang et al., teixobactin analogues, *Molecular Pharmaceutics* (2021), DOI `10.1021/acs.molpharmaceut.0c00990` | Equilibrium/free-flowing aqueous solubility combined with turbidity and viscosity screening | Analogue series centered on teixobactin | Directly addresses the boundary between dissolution and gelation; valuable for the difficulty taxonomy | Endpoint deliberately combines molecular dissolution with acceptable flow/gelation; likely small and scaffold-specific; supporting data are PDF-based |
| E004 | Trevisani et al., *Science Advances* (2025), DOI `10.1126/sciadv.adq2616` | LC-MS recovery after diluting three synthesized linear peptides to 25 micromolar in water or DMEM and incubating for one hour at 37 degrees C | Three experimentally measured peptides; a separate 16-peptide training panel contains eight `soluble` and eight `insoluble` labels | Canonical short linear peptides and an explicit condition-sensitive test | The training panel was selected from the Antimicrobial Peptide Database; its supplementary row table gives no direct solubility measurement, assay condition or row-level primary citation and is therefore ineligible |
| E008 | stapled-peptide design-rules dataset, *Nature Communications* (2024), DOI `10.1038/s41467-023-43346-4` | Kinetic solubility after dilution from DMSO into PBS at pH 7, filtration and UPLC/DAD | 234 exact numerical values among 418 analogue rows; 139 missing and 45 entries recorded as `Poorly Behaved` | Largest directly audited numerical peptide-solubility collection in this pass | Close modified/stapled analogue series; canonical pLM input omits staples, D-residues, terminal chemistry and other non-natural features |
| E005 | Do et al., *RSC Advances* (2019), DOI `10.1039/c9ra05730g` | Equilibrium water solubility of glycine- or alanine-based dipeptides across temperature, using gravimetric or photometric methods | Small panel of dipeptides with repeated measurements | High-quality thermodynamic measurements and solid-state interpretation | Too short and too small for a meaningful pLM benchmark; narrow chemistry |
| E006 | Lyu et al., ChemBioChem (2026), DOI `10.1002/cbic.70329` | Thermodynamic solubility of selected cyclic heptapeptides in PBS after 24 h, quantified by UPLC-MS | Only a selected active subset plus mortiamide-related stereochemical analogues | Strong evidence that stereochemistry and scaffold conformation can dominate solubility despite similar composition | Small, deliberately selected and rich in non-natural stereochemistry; not directly encodable by canonical ProtT5/ESM-2 tokens |
| E007 | Adler-Abramovich/related aromatic-peptide study, *Nature Communications* (2021), PMCID `PMC8063180` | Critical soluble concentration of aromatic di- and tripeptides | Small aromatic series | Mechanistic evidence that solid-state packing and aromatic interactions can reverse simple hydrophobicity expectations | Very small and assembly-coupled; unsuitable as a stand-alone predictive benchmark |

All entries remain **discovery candidates** until the primary article,
supplementary files and row-level data are opened and verified by an accountable
human. Search snippets and generated summaries are not final evidence.

## Explicitly excluded endpoint families

The following are useful for an endpoint-audit table but are not eligible as
general-peptide aqueous-solubility benchmarks:

- PROSO II and the PeptideBERT 18,453-label set: PSI production-stage status;
- SoluProt/NESG and NetSolP: recombinant-expression soluble recovery/status;
- eSOL: cell-free-expression supernatant fraction after centrifugation;
- PLM_Sol and related integrated protein datasets: heterogeneous protein
  production labels;
- predicted CamSol or other model scores without a direct experimental value;
- aggregation, self-assembly, turbidity or chromatographic hydrophobicity used
  alone as a solubility label.

## Where peptide-solubility prediction becomes difficult

The preliminary sources support a richer scientific story than a single average
accuracy value. The following strata are hypotheses to operationalize using
development data, not labels to assign by intuition alone.

### 1. Classically apparent cases

Several low-cost cues agree: ionization at the assay pH, charge density,
hydrophobic burden, aromatic content, peptide size and absence of a concentrated
hydrophobic segment. A transparent physicochemical model may already be
sufficient. The benchmark should test whether a pLM produces a practically
important paired improvement in this stratum rather than assuming that greater
model complexity is useful.

### 2. Cue-conflicted cases

Strong cues point in opposing directions. Examples include a highly charged
peptide with an aromatic or aliphatic aggregation-prone patch, or a globally
hydrophilic peptide with a locally ordered self-association motif. These are
plausible candidates for contextual sequence representations, provided the
training data contain relevant examples.

### 3. Sequence- or scaffold-context cases

Residue order, D/L stereochemistry, N-methylation, cyclization and conformational
preferences can alter exposure of polar groups and solid-state packing without a
large change in bulk composition. Canonical protein pLMs do not natively encode
several of these chemical distinctions. A pLM failure here may be an input-
representation failure rather than evidence that context is unimportant.

### 4. Condition-sensitive cases

Solubility can change with pH, buffer, salt, temperature, incubation, starting
solvent, solid form and concentration protocol. A sequence-only model cannot
recover unrecorded conditions. These cases require condition-aware modelling or
separate endpoints rather than a larger sequence encoder.

### 5. Phase-boundary cases

Gelation, colloidal aggregation, precipitation, LLPS and ordered self-assembly can
make the measured supernatant concentration or flow behavior protocol-dependent.
The teixobactin assay is informative precisely because it separates visual
dissolution from acceptable viscosity. These observations must not be reduced to
an undefined binary 'soluble' label.

### 6. Unsupported chemistry

Non-natural residues, terminal conjugates, cyclic topology and unusual salt forms
may fall outside both the benchmark and the pLM vocabulary. Applicability-domain
and chemical-identity checks must precede interpretation.

## Benchmark gate

No embeddings or predictive models should be fitted until a row-level audit can
answer all of the following:

1. How many unique chemical peptide entities remain?
2. How many have directly measured values with units and traceable conditions?
3. Can measurements be harmonized without erasing assay meaning?
4. How many independent scaffolds or sequence families exist?
5. Is there a canonical linear subset large enough for ProtT5/ESM-2?
6. Can all close analogues and scaffold relatives be kept in the same split?
7. Is there an external source or assay suitable for transfer evaluation?

Working decisions:

- at least 500 diverse, traceable peptides: potential strong GO;
- roughly 150--499 across several independent scaffolds/sources: conditional GO
  for a restrained proof of concept;
- fewer than about 150, three or fewer dominant scaffolds, or irreconcilable
  condition heterogeneity: NO-GO for a general benchmark.

## Completed audit actions

1. CamSol-PTM source and supplementary workbooks were downloaded, checksummed
   and audited.
2. The PEGASUS repository was pinned to commit
   `5df3a4b073181b08175af16dc7e0526aa5e34179`; direct-solubility rows and
   chemical-string collisions were audited.
3. The stapled-peptide workbook was audited across three code-verified table
   sections; exact, missing and categorical textual values were counted separately.
4. The correct 45-page Science Advances supplement was retrieved from the NCBI
   open-access package and its 16-row training table was inspected directly.
5. Database-specific discovery searches were saved under `search_results/`.
6. A formal decision was issued before any new model, manuscript or graphical
   abstract was created.
