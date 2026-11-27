# Search log — general-peptide aqueous solubility

**Access date:** 4 August 2026
**Stage:** reproducible discovery search and targeted file audit; not exhaustive
and not PRISMA-complete

## Scope lock

Included discovery target: experimentally measured aqueous solubility of a
chemically defined peptide. Recombinant expression, production-stage labels,
predicted scores and aggregation-only proxies were excluded from benchmark
eligibility.

## Web queries run

1. `"peptide aqueous solubility" dataset experimental sequence`
2. `"intrinsic solubility" peptide dataset sequence`
3. `"apparent solubility" peptides dataset sequence`
4. `"cyclic peptide" solubility dataset experimental`
5. `"PEGASUS" "aqueous solubility" cyclic peptides supplementary data`
6. `"aqueous solubility assay" "cyclic peptides" HBSS dataset`
7. `site:nature.com cyclic peptide solubility assay dataset supplementary data 2025 2026`
8. `"peptide solubility" "Source Data" aqueous`
9. `PepSeA peptide solubility dataset experimental`
10. `"peptide solubility" database sequence measured`
11. `"aqueous solubility" "peptide" machine learning dataset`
12. `"peptide solubility prediction" experimental dataset`
13. `"high-throughput" peptide "aqueous solubility"`
14. `"peptide library" "aqueous solubility assay"`
15. `"intrinsic solubility" "peptide variants"`
16. `"solubility" "cyclic peptide library" LC-MS`
17. `dipeptide aqueous solubility dataset all dipeptides`
18. `experimental solubility oligopeptides water sequence`
19. `"solubilities of dipeptides" water`
20. `"aqueous solubility" tripeptides`

Targeted title/identifier searches were also run for the PEGASUS, CamSol-PTM,
claudin-5 peptide, mortiamide/cyclic-heptapeptide and Science Advances
supplementary records.

## Primary landing pages discovered

- CamSol-PTM: https://www.nature.com/articles/s41467-023-42940-w
- PEGASUS: https://doi.org/10.1021/acs.jmedchem.5c01836
- Teixobactin solubility assay: https://doi.org/10.1021/acs.molpharmaceut.0c00990
- Claudin-5 peptide study: https://doi.org/10.1126/sciadv.adq2616
- Dipeptide thermodynamic solubility: https://doi.org/10.1039/c9ra05730g
- Cyclic-heptapeptide/mortiamide study: https://doi.org/10.1002/cbic.70329

## Limitations of this search pass

- Search-result snippets were used for discovery only.
- The most relevant CamSol-PTM, PEGASUS, stapled-peptide and claudin-5 public
  files were downloaded and row/endpoint-audited, but the complete broad search
  set was not screened as a systematic review.
- Database-specific result counts were produced, but no deduplicated PRISMA
  count was claimed.
- Google Patents and vendor pages were not accepted as primary benchmark sources.
- Absence from this preliminary search is not evidence that a dataset does not
  exist.

The independent Deep Research pass must provide multi-database counts,
deduplication and explicit full-text exclusion reasons.

## Reproducible database queries

The following queries were run on 4 August 2026 and their raw JSON responses
were retained under `search_results/`.

| Database | Query | Returned count | Stored file | Interpretation |
|---|---|---:|---|---|
| PubMed | `peptide*[Title/Abstract] AND ("aqueous solubility"[Title/Abstract] OR "intrinsic solubility"[Title/Abstract] OR "kinetic solubility"[Title/Abstract])` | 272 | `pubmed_direct_solubility.json` | Broad discovery set, not an eligibility count |
| PubMed | `peptide*[Title/Abstract] AND solubility[Title/Abstract] AND ("protein language model"[Title/Abstract] OR ProtT5[Title/Abstract] OR ESM-2[Title/Abstract] OR ProtBERT[Title/Abstract])` | 3 | `pubmed_plm_solubility.json` | Narrow pLM query; includes reused operational labels and is not evidence of direct aqueous-solubility data |
| Europe PMC | `TITLE_ABS:peptide* AND (TITLE_ABS:"aqueous solubility" OR TITLE_ABS:"intrinsic solubility" OR TITLE_ABS:"kinetic solubility")` | 302 | `europepmc_direct_solubility.json` | Broad discovery set including preprints and indexed variants |
| OpenAlex | full-text search for `peptide aqueous solubility dataset prediction` | 8,057 | `openalex_direct_solubility.json` | Too nonspecific for screening; the first 200 records were retained to document query failure |

The PubMed pLM query returned PMIDs `42232867`, `41675712` and `37956397`.
Direct record checking identified these as a gated-query-pooling peptide-property
study, a general efficient-pLM architecture paper, and PeptideBERT,
respectively. The PeptideBERT-related solubility endpoint is inherited from
production-stage labels rather than direct peptide aqueous-solubility assays.

## File-level audits completed

- CamSol-PTM: `camsolptm_endpoint_audit_20260804.json`
- PEGASUS: `pegasus_endpoint_audit_20260804.json`
- stapled peptides: `stapled_peptides_endpoint_audit_20260804.json`
- claudin-5 supplement: `derived/sciadv.adq2616_sm.txt` and the checksummed
  source PDF `raw/sciadv.adq2616_sm.pdf`

The numerical audit outputs contain aggregate information only. They do not
redistribute raw peptide rows or imply that heterogeneous measurements can be
combined.
