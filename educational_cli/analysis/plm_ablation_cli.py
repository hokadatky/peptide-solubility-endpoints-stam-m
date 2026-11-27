#!/usr/bin/env python
"""Apply the STAM-M representation-ablation method to YOUR OWN peptide data.

Given a CSV of (id, sequence, label[, group]), this asks the paper's question on
your endpoint: does adding a protein language model (ProtT5 / ESM-2) to a
transparent baseline of amino-acid composition + physicochemical descriptors
improve prediction under a homology-controlled split, judged by a paired cluster
bootstrap against a prespecified margin?

Example
-------
    # BASE-only smoke test (no torch needed):
    python analysis/plm_ablation_cli.py --input examples/peptides_demo.csv \
        --features base --outdir out_demo

    # full comparison (needs the 'prot5' env: torch + transformers):
    python analysis/plm_ablation_cli.py --input mydata.csv \
        --id-col id --seq-col sequence --label-col label --group-col scaffold \
        --features base,prott5,esm2 --pls-components 3 --margin 0.03 \
        --outdir out_mydata

Interpretation
--------------
Each pLM arm is reported as a paired delta vs BASE with a 95% cluster-bootstrap
interval and a within-margin flag. Read a null against BASE, not against zero:
an interval inside +/- margin means "no additive gain at this sample size", NOT
"equivalent". Small n gives wide intervals (that is the honest answer). Canonical
one-letter tokens do not represent staples, D-residues, N-methylation or
cyclisation -- for such chemistry a sequence-only gain is not expected.

Outputs aggregate JSON only; your sequences/labels are never copied into it.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ablation_core as core  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with id, sequence, label[, group]")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default=None,
                    help="optional homology/scaffold cluster column; if absent, "
                         "sequences are greedily clustered by k-mer similarity")
    ap.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    ap.add_argument("--features", default="base,prott5,esm2",
                    help="comma list: base,prott5,esm2 (base is always included)")
    ap.add_argument("--pls-components", type=int, default=0,
                    help=">0 => deploy each pLM as a k-component supervised PLS "
                         "projection (the low-dimensional form); 0 => raw concat")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="prespecified practical margin (AUC or log-RMSE units)")
    ap.add_argument("--cluster-threshold", type=float, default=0.6,
                    help="k-mer Jaccard threshold for fallback clustering")
    ap.add_argument("--cache-dir", default=".plm_cache")
    ap.add_argument("--outdir", default="ablation_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for c in (args.id_col, args.seq_col, args.label_col):
        if c not in df.columns:
            ap.error(f"column '{c}' not in {args.input} (have: {list(df.columns)})")

    raw_seqs = df[args.seq_col].astype(str).tolist()
    seqs = [core.clean_sequence(s) for s in raw_seqs]
    dropped_chars = sum(len(core.clean_sequence(a)) != len(str(a).strip())
                        for a in raw_seqs)
    keep = [i for i, s in enumerate(seqs) if len(s) >= 2]
    if len(keep) < len(seqs):
        print(f"[warn] dropped {len(seqs)-len(keep)} rows with <2 canonical residues")
    seqs = [seqs[i] for i in keep]
    y = pd.to_numeric(df[args.label_col].iloc[keep], errors="coerce").to_numpy()
    ok = ~np.isnan(y)
    if not ok.all():
        print(f"[warn] dropped {int((~ok).sum())} rows with non-numeric label")
        seqs = [s for s, k in zip(seqs, ok) if k]
        y = y[ok]
    group = None
    if args.group_col:
        if args.group_col not in df.columns:
            ap.error(f"group column '{args.group_col}' not found")
        group = df[args.group_col].iloc[keep].to_numpy()[ok]

    task = core.detect_task(y) if args.task == "auto" else args.task
    if task == "classification":
        y = (y > 0).astype(int) if set(np.unique(y)) - {0, 1} else y.astype(int)

    groups = core.make_groups(seqs, group=group, threshold=args.cluster_threshold)
    n, ng = len(seqs), len(np.unique(groups))
    print(f"[info] n={n}  clusters={ng}  task={task}  "
          f"{'(explicit groups)' if group is not None else '(k-mer fallback clusters)'}")
    if dropped_chars:
        print(f"[info] {dropped_chars} sequences contained non-canonical characters "
              f"that were dropped (modified/cyclic chemistry is not represented)")

    want = [w.strip().lower() for w in args.features.split(",")]
    Xbase = core.base_features(seqs)

    metric = "AUC" if task == "classification" else "RMSE_log_or_native"
    result = {"n": n, "n_clusters": int(ng), "task": task, "metric": metric,
              "margin": args.margin, "pls_components": args.pls_components,
              "folds": args.folds, "repeats": args.repeats, "bootstrap": args.bootstrap,
              "arms": {}}

    # BASE
    base_oof, _ = core.oof_predictions(Xbase, y, groups, task, n_splits=args.folds,
                                       repeats=args.repeats, seed=args.seed)
    result["arms"]["BASE"] = {"score": core._metric(y, base_oof, task)}
    print(f"[BASE] {metric} = {result['arms']['BASE']['score']:.4f}")

    blocks = {}
    if "prott5" in want:
        blocks["ProtT5"] = ("prott5", None)
    if "esm2" in want:
        blocks["ESM2"] = ("esm2", None)

    embcache = {}
    for name, (which, _mid) in blocks.items():
        from embeddings import embed
        emb = embcache.get(which)
        if emb is None:
            print(f"[info] computing {name} embeddings (cache={args.cache_dir}) ...")
            emb = embed(seqs, which, cache_dir=args.cache_dir)
            embcache[which] = emb
        oof, _ = core.oof_predictions(Xbase, y, groups, task, pls_block=emb,
                                      pls_k=args.pls_components, n_splits=args.folds,
                                      repeats=args.repeats, seed=args.seed)
        d, lo, hi = core.paired_cluster_bootstrap(y, groups, base_oof, oof, task,
                                                  n_boot=args.bootstrap, seed=args.seed)
        # for RMSE, "improvement" is negative delta; normalize sign of interest
        within = (abs(d) <= args.margin) and (abs(lo) <= args.margin) and (abs(hi) <= args.margin)
        arm = {"score": core._metric(y, oof, task),
               "delta_vs_base": d, "ci95": [lo, hi], "within_margin": bool(within)}
        result["arms"][f"BASE+{name}"] = arm
        print(f"[BASE+{name}] {metric} = {arm['score']:.4f}  "
              f"delta = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'within margin (no gain)' if within else 'outside margin'}")

    os.makedirs(args.outdir, exist_ok=True)
    outp = os.path.join(args.outdir, "ablation_metrics.json")
    with open(outp, "w") as h:
        json.dump(result, h, indent=2)
    print(f"[done] wrote {outp}")
    if task == "regression":
        print("[note] lower RMSE is better; a negative delta means the pLM helps.")


if __name__ == "__main__":
    main()
