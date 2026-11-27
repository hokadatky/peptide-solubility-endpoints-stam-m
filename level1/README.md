# Level 1: figures and tables from aggregate records

Everything needed to regenerate the article's figures, graphical abstract and Supplementary Tables
S3, S7 and S8 is in this folder (Tables S7 and S8 also read their result files from `../level3/`).
Run the commands in the top-level README from here; outputs go to `figures/` and `tables/`.

| Folder | Contents |
|---|---|
| `scripts/` | R figure scripts, table generators, the scripts that released the aggregate records, and the public-data audit scripts |
| `analysis_addon/results/` | Aggregate records read by the figure and table scripts; `pepsol_explorer_s2_demo.json` is a static synthetic record for Figure 3 only |
| `audit/` | Search log and search responses, feasibility audit and go/no-go record, aggregate endpoint audits of the direct-peptide sources |
