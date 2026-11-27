# Redaction notice — raw search responses

These four files are the raw responses from the database searches used to scope
the review. They are published as query provenance, not as an eligibility count
and not as a substitute for the primary literature.

One of them was redacted before publication.

## `openalex_direct_solubility.json`

OpenAlex returns each work's corresponding-author contact details inside the
work record. The response therefore carried **113 distinct third-party email
addresses** belonging to authors who have nothing to do with this article and
who did not consent to appear in this repository.

Every email-like string in this file was replaced with the literal

```
[redacted-third-party-email]
```

Nothing else was altered: identifiers, DOIs, titles, venue, year, counts,
authorship names, the result ordering and the `meta` block are untouched.

**No reproducibility is lost.** The exact query is recorded in the file itself
at `meta.x_query.url`:

```
/works?filter=fulltext.search:peptide aqueous solubility dataset prediction&per_page=200
```

Anyone can re-issue that query against the OpenAlex API and obtain the
unredacted response directly from the source, which is the correct place to get
third-party contact details if they are ever needed.

The redaction is performed by the `REGEX_SANITISE` rule in
`tools/build_public_payload.py`, so it is reproducible and fails loudly if the
pattern ever stops matching.

## `pubmed_direct_solubility.json`, `pubmed_plm_solubility.json`, `europepmc_direct_solubility.json`

Scanned and **not** redacted: these responses contain identifiers, counts and
bibliographic fields only, with no contact details.
