#!/usr/bin/env python3
"""Generate Supplementary Table S7 from the stored external-evaluation results.

Inputs (read-only):
  paper/stam_nesg_external_20261016/results/external_retrospective.json
      the prespecified B1 versus B1+ProtT5 (supervised PLS) external comparison
  paper/stam_nesg_external_20261016/exploratory_improvement/results.json
      the three exploratory representation/head comparisons specified after the
      first external result was seen

Output:
  tables/supp_table_s7_external.tex

Numeric literals are never edited by hand; rerun this script instead.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
RUN1 = ROOT.parent / "level3/04_external_evaluation/nesg/results/external_retrospective.json"
RUN2 = ROOT.parent / "level3/04_external_evaluation/nesg/exploratory_improvement/results.json"
OUT = ROOT / "tables/supp_table_s7_external.tex"


def f(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 4) -> str:
    return f"$+{x:.{digits}f}$" if x >= 0 else f"$-{abs(x):.{digits}f}$"


def num(n: int) -> str:
    """Integer with a TeX-safe thousands separator."""
    return f"{n:,}".replace(",", "{,}")


def interval(lo: float, hi: float, digits: int = 4) -> str:
    def part(v: float) -> str:
        return f"+{v:.{digits}f}" if v >= 0 else f"-{abs(v):.{digits}f}"

    return f"[${part(lo)}$, ${part(hi)}$]"


def main() -> int:
    run1 = json.loads(RUN1.read_text())
    run2 = json.loads(RUN2.read_text())

    a1 = run1["arms"]["B1_ORDER_LOGIT"]["calibrated"]["metrics"]
    a2 = run1["arms"]["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["calibrated"]["metrics"]
    pls_k = run1["arms"]["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["selected_k_C"][0]
    pc = run1["primary_contrast"]

    arms = run2["arms"]
    contrasts = {c["candidate"]: c for c in run2["uncertainty"]["contrasts"]}

    def m(key: str) -> dict:
        return arms[key]["calibrated"]

    def sel(key: str) -> str:
        s = arms[key]["selected"]
        uncompressed = key == "FULL_LOGIT"
        if "C" in s:
            text = f"$C={s['C']:g}$"
            if uncompressed:
                return "all 1{,}024 dimensions, " + text
            if s.get("k"):
                text = f"{s['k']} components, " + text
            return text
        bits = [
            f"{s['max_leaf_nodes']} leaves",
            f"min.\\ {s['min_samples_leaf']} rows",
            f"$L_2={s['l2_regularization']:g}$",
        ]
        if s.get("k"):
            bits.insert(0, f"{s['k']} components")
        return ", ".join(bits)

    rows_pre = [
        ("B1, logistic (reference)", f"$C={run1['arms']['B1_ORDER_LOGIT']['selected_k_C'][1]:g}$",
         a1["roc_auc"], a1["auprc"], a1["brier"], a1["log_loss"]),
        ("B1 + ProtT5, supervised PLS, logistic",
         f"{pls_k} components, $C={run1['arms']['B1_ORDER_PLUS_PROTT5_PLS_LOGIT']['selected_k_C'][1]:g}$",
         a2["roc_auc"], a2["auprc"], a2["brier"], a2["log_loss"]),
    ]

    rows_exp = [
        ("B1 + ProtT5, uncompressed, logistic", sel("FULL_LOGIT"),
         m("FULL_LOGIT")["auc"], m("FULL_LOGIT")["average_precision"],
         m("FULL_LOGIT")["brier"], m("FULL_LOGIT")["log_loss"]),
        ("B1 + ProtT5, PCA, logistic", sel("PCA_LOGIT"),
         m("PCA_LOGIT")["auc"], m("PCA_LOGIT")["average_precision"],
         m("PCA_LOGIT")["brier"], m("PCA_LOGIT")["log_loss"]),
        ("B1, gradient boosting (reference)", sel("B1_HGB"),
         m("B1_HGB")["auc"], m("B1_HGB")["average_precision"],
         m("B1_HGB")["brier"], m("B1_HGB")["log_loss"]),
        ("B1 + ProtT5, PCA, gradient boosting", sel("PCA_HGB"),
         m("PCA_HGB")["auc"], m("PCA_HGB")["average_precision"],
         m("PCA_HGB")["brier"], m("PCA_HGB")["log_loss"]),
    ]

    def arm_row(r) -> str:
        name, setting, auc, ap, brier, ll = r
        return (f"{name} & {setting} & {f(auc)} & {f(ap)} & {f(brier)} & {f(ll)} \\\\")

    # Development reference for the five exploratory arms. Both columns come
    # from the same fitted model: the raw-score ROC-AUC on the development
    # partition that was also used to fit the Platt calibration, and the
    # development-calibrated external ROC-AUC. The development column is not
    # the canonical grouped seven-fold out-of-fold estimate reported in the
    # main text, and the two columns describe different populations.
    dev_rows = [
        ("B1, logistic", "B1_LOGIT"),
        ("B1 + ProtT5, uncompressed, logistic", "FULL_LOGIT"),
        ("B1 + ProtT5, PCA, logistic", "PCA_LOGIT"),
        ("B1, gradient boosting", "B1_HGB"),
        ("B1 + ProtT5, PCA, gradient boosting", "PCA_HGB"),
    ]
    for _, key in dev_rows:
        if "calibration_partition_raw_metrics" not in arms[key]:
            raise SystemExit(
                f"error: no stored development calibration-partition metrics for {key}; "
                "do not substitute another estimate."
            )

    contrast_rows = [
        ("B1 + ProtT5 (supervised PLS) $-$ B1, logistic",
         signed(pc["point_gain"]), interval(*pc["ci95"]), "not applicable"),
    ]
    for key, label in (
        ("FULL_LOGIT", "B1 + ProtT5 (uncompressed) $-$ B1, logistic"),
        ("PCA_LOGIT", "B1 + ProtT5 (PCA) $-$ B1, logistic"),
        ("PCA_HGB", "B1 + ProtT5 (PCA) $-$ B1, gradient boosting"),
    ):
        c = contrasts[key]
        contrast_rows.append(
            (label, signed(c["delta_auc"]), interval(*c["ci95"]),
             interval(*c["bonferroni_ci_98_333333"]))
        )

    n_rows = run1["n_rows"]
    n_groups = run1["n_groups"]
    pos = run1["positive"]
    reps = pc["n_replicates_valid"]

    body = []
    body.append("% Generated by scripts/make_supp_table_s7_external.py; "
                "do not edit numeric literals by hand.")
    body.append(f"% Sources: {RUN1.name} (prespecified contrast), "
                f"{RUN2.parent.name}/{RUN2.name} (exploratory contrasts).")
    body.append(r"\begin{table}[H]")
    # The whole table is new in revision 1, so the highlighted build marks the
    # caption, body and notes together. \revall is a color switch scoped to
    # this float; see the preamble note in Supplementary_Information_S2.tex.
    body.append(r"\revall")
    body.append(r"\centering")
    body.append(
        r"\caption{\rev{Retrospective external evaluation on the separate low-similarity NESG test set. "
        "All models were fitted, selected and calibrated on development data only and were scored on "
        f"{num(n_rows)} sequences in {num(n_groups)} sequence-identity groups ({num(pos)} soluble, "
        f"{num(run1['negative'])} insoluble). "
        + r"The first block repeats the prespecified comparison; the second reports the three "
          r"exploratory comparisons specified after that result was seen; the third places the "
          r"development and external ROC-AUC of the same five fitted exploratory models side by "
          r"side. Every comparison performed "
          r"is listed, including the unfavorable one. Probabilities are development-calibrated; "
          r"no external outcome was used for tuning, selection or recalibration.}}"
    )
    body.append(r"\label{tab:s7-external}")
    body.append(r"\fontsize{8.4}{10.4}\selectfont")
    body.append(r"\renewcommand{\arraystretch}{1.16}")
    body.append(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X "
                r">{\raggedright\arraybackslash}p{0.21\textwidth} r r r r}")
    body.append(r"\toprule")
    body.append(r"\textbf{Model} & \textbf{Selected setting} & \textbf{ROC-AUC} & "
                r"\textbf{AP} & \textbf{Brier} & \textbf{Log loss} \\")
    body.append(r"\midrule")
    body.append(r"\multicolumn{6}{l}{\emph{Prespecified comparison}} \\")
    body.extend(arm_row(r) for r in rows_pre)
    body.append(r"\midrule")
    body.append(r"\multicolumn{6}{l}{\emph{Exploratory comparisons, specified after the first "
                r"external result}} \\")
    body.extend(arm_row(r) for r in rows_exp)
    body.append(r"\bottomrule")
    body.append(r"\end{tabularx}")
    body.append("")
    body.append(r"\vspace{6pt}")
    body.append(r"\begin{tabular}{l r l l}")
    body.append(r"\toprule")
    body.append(r"\textbf{Paired contrast (ROC-AUC)} & \textbf{Difference} & "
                r"\textbf{95\% CI} & \textbf{Adjusted CI} \\")
    body.append(r"\midrule")
    for label, delta, ci, adj in contrast_rows:
        body.append(f"{label} & {delta} & {ci} & {adj} \\\\")
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append("")
    body.append(r"\vspace{6pt}")
    body.append(r"\begin{tabular}{l r r}")
    body.append(r"\toprule")
    body.append(r"\textbf{Exploratory arm, same fitted model} & "
                r"\textbf{Development ROC-AUC} & \textbf{External ROC-AUC} \\")
    body.append(r" & \emph{calibration partition, raw} & \emph{calibrated} \\")
    body.append(r"\midrule")
    for label, key in dev_rows:
        dev = arms[key]["calibration_partition_raw_metrics"]["auc"]
        ext = arms[key]["calibrated"]["auc"]
        body.append(f"{label} & {f(dev)} & {f(ext)} \\\\")
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")
    body.append("")
    body.append(r"\vspace{4pt}")
    body.append(r"\begin{minipage}{0.96\textwidth}")
    body.append(r"\footnotesize")
    body.append(
        f"Intervals use {num(reps)} paired cluster-bootstrap resamples with the sequence-identity "
        "group as the unit. "
        + r"The adjusted interval is a Bonferroni 98.3333\% interval covering the three "
          r"exploratory contrasts only; it does not adjust for earlier method development or for "
          r"prior exposure to these external outcomes. All intervals condition on the fitted "
          r"predictions and exclude refitting, method-selection and between-study uncertainty. "
          r"The logistic B1 reference is the same fitted model in every block. "
        + f"In the third block the development column is the raw-score ROC-AUC of that same "
          f"fitted model on the {num(run1['arms']['B1_ORDER_LOGIT']['calibration_n'])} development "
          r"records that were also used to fit its Platt calibration. It is a descriptive "
          r"within-development reference, not the canonical grouped seven-fold out-of-fold "
          r"estimate reported in the main article and not an independent confirmatory estimate. "
          r"The two columns describe different populations, so no paired test is applied across "
          r"them. Secondary metrics "
          r"are descriptive and were not tested. This evaluation is retrospective: outcomes from "
          r"this test set had been examined during earlier method development, so it is not an "
          r"untouched confirmatory test set."
    )
    body.append(r"\end{minipage}")
    body.append(r"\end{table}")

    OUT.write_text("\n".join(body) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
