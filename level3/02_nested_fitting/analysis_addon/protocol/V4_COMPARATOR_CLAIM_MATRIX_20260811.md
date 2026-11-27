# V4 comparator and claim-commensurability audit

Date: 2026-08-11 (Asia/Tokyo)  
Status: literature side frozen; V4 result cells pending canonical aggregation  
Scope: claim control for the retrospective SoluProt worked example

## Decision rule

An accuracy ranking is permitted only when endpoint, source rows, exclusion
rules, train/test homology policy, grouping unit, metric, and test-label access
are the same. A model reported on a different benchmark is context, not a
leaderboard comparator. In particular, the following are not interchangeable:

- TargetTrack/NESG recombinant soluble-expression status;
- eSOL cell-free soluble-fraction measurements;
- UESolDS or PDB-derived production-stage labels;
- protocol-conditioned soluble recovery;
- direct aqueous concentration for a chemically defined peptide.

## Frozen V4 row

| Field | V4 contract |
|---|---|
| Endpoint | Binary recombinant soluble-expression status in *E. coli*; not direct aqueous solubility |
| Development data | 11,436 TargetTrack/SoluProt-derived rows; 10,715 existing 30%-identity groups |
| External data | 3,100 retrospective NESG rows; 2,723 external groups |
| Primary contrast | B1 order-aware transparent baseline versus B1 + fold-local ProtT5 PLS |
| Internal evaluation | Seven-fold group-exclusive nested OOF ROC-AUC; paired group bootstrap |
| External procedure | Development-only fit/selection/refit/calibration seal, then one retrospective application |
| Status | Numerical result pending canonical V4 aggregation |

## Current comparison landscape

| Work | Reported task/evidence | Why it is not a like-for-like V4 ranking |
|---|---|---|
| SoluProt (Hon et al., 2021; DOI 10.1093/bioinformatics/btaa1102) | TargetTrack training and balanced 3,100-sequence NESG test; published external AUC 0.62 | Closest historical comparator, but the published feature set, fitting procedure, calibration, and current V4 grouping contract differ |
| NetSolP (Thumuluri et al., 2022; DOI 10.1093/bioinformatics/btab801) | PLM-based *E. coli* expression/purification-usability prediction with identity-aware partitions | Different curated rows and split construction |
| PLM_Sol (Zhang et al., 2024; DOI 10.1093/bib/bbae404) | UESolDS classification; reported independent-test AUC 0.8342 | Different integrated dataset, labels, exclusions, and test set |
| SurfSol (Fang et al., 2026; DOI 10.1021/acs.jcim.6c00124) | Processed eSOL; reported R2 0.555 and AUC 0.895 | eSOL soluble fraction, structure/surface inputs, and independent split are not the V4 endpoint or rows |
| FGNNSol (Song et al., 2026; DOI 10.1021/acs.jcim.5c02262) | eSOL regression with AlphaFold3/ESM-C; reported eSOL R2 0.577 | Different continuous endpoint, multimodal inputs, rows, and split |
| ProtSATT (Deng et al., 2026; DOI 10.1021/acs.jcim.6c00821) | eSOL, *S. cerevisiae*, and *E. coli* regression/classification benchmarks using three PLMs | Multiple different benchmark contracts; not rerun on the V4 rows |
| DeepSolNet (Cui et al., 2026; DOI 10.3389/fmicb.2026.1716930) | ESM-C model on UESolDS and an additional literature-derived test | Different data construction and test set |
| MPRL (Yang et al., 2026; DOI 10.1016/j.eswa.2026.131142) | Multi-perspective recombinant-protein solubility modelling | Different model inputs, data, and split |
| ProSolNet/ProSolNet-Mut (Wang et al., 2026; DOI 10.1186/s13321-026-01225-2) | Solubility classification and mutation-effect prediction using multimodal structure/surface features | Different tasks, modalities, and benchmarks |
| SeqStruct-XAttn (Elmi, 2026; DOI 10.1016/j.compbiomed.2026.111821) | eSOL independent test; reported ensemble AUC 0.920 | Different continuous/binary derivation, rows, structure inputs, and split |
| Aiki-Sol (Rajagopalan et al., 2026 preprint; DOI 10.64898/2026.05.14.725067) | Protocol-conditioned *E. coli* soluble recovery; reported five-cohort mean AUC 0.825 | Different, much larger curated training pool and protocol-conditioned outputs; preprint, not the V4 contract |

## Claim ceiling

- V4 can test whether ProtT5 adds value to the frozen transparent baseline under
  the V4 internal and retrospective external contracts.
- V4 cannot establish the world's best aqueous-solubility predictor because it
  does not measure direct aqueous solubility and does not encode full peptide
  chemistry.
- V4 cannot claim the best recombinant-expression predictor unless named
  comparators are rerun on the identical rows, groups, split, metric, and
  external-label policy. No such common rerun is part of the frozen V4 plan.
- Therefore all reported values from other papers remain a context table. They
  must not be plotted on one accuracy axis with V4.

## Metadata provenance

Publisher/Crossref metadata and primary abstracts were checked on 2026-08-11.
The ProSolNet version of record was assigned Journal of Cheminformatics volume
18, article 113 on 2026-08-07. Aiki-Sol was still a bioRxiv preprint at the time
of this audit.
