# Third-party data and literature records

The manuscript, original code, figures and audit commentary are author
material. This package does **not** relicense or redistribute the source
workbooks, journal supplements or PEGASUS row-level data.

## Audited public sources

| Source | Persistent identifier | Audited local source SHA-256 | Redistribution decision |
|---|---|---|---|
| CamSol-PTM supplementary sequence table | `10.1038/s41467-023-42940-w` | `400232664ce3d67bd349659a04d5f319fadd7ba50e76ca278f691560c36d97b5` | Raw workbook excluded |
| CamSol-PTM Figure 3 source data | `10.1038/s41467-023-42940-w` | `8ae367129d0e6aa9ceedb7afbd4a2b38469b8f121aae4e8b4ac2de91582b4b1d` | Raw workbook excluded |
| Stapled-peptide Supplementary Data 1 | `10.1038/s41467-023-43346-4` | `cc814db9acd610018f8cf4d6b030a95fb5a3659aa49bb3644c0883700a637553` | Raw workbook excluded |
| PEGASUS `inhouse_design.csv` derivative used locally | `10.1021/acs.jmedchem.5c01836`; upstream commit `5df3a4b073181b08175af16dc7e0526aa5e34179` | Normalised local input: `8f0f71a8cebc2a86dd5cf17a0be16e1bf8bbfd6bc011cde949cb0ee6bbea7f6c` | Raw and transformed rows excluded; upstream licence is CC BY-NC-ND 4.0 |

Only aggregate JSON audits, checksums and non-row-level conclusions are
included. Publication is limited to derived counts and prespecified coarse bins:
no source files, row-level measurements or exact extreme values are
redistributed. Observed minima and maxima were removed on 5 August 2026 because
each reproduces a single third-party row verbatim, which is a weaker basis under
the CC BY-NC-ND 4.0 no-derivatives clause than an aggregate count and is not
required by any claim in the manuscript. Users must obtain each source directly from its publisher or
upstream repository and comply with the applicable terms.

Database search responses under `audit/search_results/` are retained for query
provenance. They were used for discovery, not as an eligibility count or as a
substitute for the primary literature.
