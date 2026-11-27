# Educational bring-your-own-data workflow

This directory contains a bring-your-own-data educational CLI and its small
support modules. It is separate from the manuscript's evidence and is not a
pretrained or production system.

## Input and synthetic demo

`analysis/plm_ablation_cli.py` accepts a table with `id, sequence, label[, group]`. The shipped
`../examples/peptides_demo.csv` file is synthetic and is intended only for
pipeline validation.

```bash
python analysis/plm_ablation_cli.py --input examples/peptides_demo.csv \
    --group-col scaffold --features base --outdir out_demo
```

User-generated runs must define the endpoint, grouping scheme, comparator and
validation plan independently. Do not treat a CLI output as a result of this
review or as evidence for an aqueous-solubility claim.

The CLI is not a pretrained predictor, a V4 model, a validated aqueous-solubility application, a production service, or evidence of SOTA performance. Any user-generated result requires an independently justified endpoint, grouping scheme, comparator and validation plan.

