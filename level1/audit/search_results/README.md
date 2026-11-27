# Database search responses

These are the raw responses of the literature and database searches recorded in
`../SEARCH_LOG_20260804.md`. They are kept as query provenance only; they were used for discovery,
not as an eligibility count or as a substitute for the primary literature.

| File | Service | Contents as published |
|---|---|---|
| `pubmed_direct_solubility.json`, `pubmed_plm_solubility.json` | PubMed E-utilities (esearch) | Query and matching PMIDs |
| `europepmc_direct_solubility.json` | Europe PMC REST API | Query and bibliographic metadata (no abstracts) |
| `openalex_direct_solubility.json` | OpenAlex API | Query and bibliographic metadata. E-mail-like strings were redacted (`REDACTION.md`), and the abstract field (`abstract_inverted_index`) was removed from every record before publication. |

The responses remain subject to the terms of the respective services; see `../../../docs/LICENSE-CC-BY-4.0.md`.
