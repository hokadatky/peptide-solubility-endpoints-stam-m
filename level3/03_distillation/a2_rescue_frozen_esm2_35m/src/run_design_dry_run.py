#!/usr/bin/env python3
"""Run three bounded rescue mechanisms on design folds 2--6 -> fold 0 only."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
V1_PLAN = PROJECT_ROOT / "paper/peptide_solubility_review_distillation_plan_20260813"
V1_RUN = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_run_20260813_v1"
PRIVATE_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2"
V1_CONFIG = V1_PLAN / "config/DISTILLATION_CONFIG_20260813.json"
LEGACY_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_plm_ablation"

sys.path.insert(0, str(V1_PLAN / "src"))
from teacher_crossfit import load_config, load_development, validate_input_hashes  # noqa: E402


SEED = 20260813
ALPHA = 0.75
TRAIN_FOLDS = (2, 3, 4, 5, 6)
EVALUATION_FOLD = 0


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Invalid prediction vector")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def load_teacher_targets(n_rows: int, train_indices: np.ndarray) -> np.ndarray:
    targets = np.full(n_rows, np.nan, dtype=float)
    for fold in TRAIN_FOLDS:
        path = V1_RUN / f"checkpoints/design/prott5/crossfit_fold_{fold}.npz"
        with np.load(path, allow_pickle=False) as item:
            indices = item["row_indices"].astype(int)
            probabilities = item["probabilities"].astype(float)
        targets[indices] = probabilities
    if not np.isfinite(targets[train_indices]).all():
        raise RuntimeError("Cross-fitted teacher targets are incomplete")
    if np.isfinite(targets[np.setdiff1d(np.arange(n_rows), train_indices)]).any():
        raise RuntimeError("Teacher target coverage exceeds the fixed design folds")
    return targets


def duplicate_fractional_targets(x, target: np.ndarray):
    y = np.concatenate([np.ones(len(target), dtype=int), np.zeros(len(target), dtype=int)])
    weight = np.concatenate([target, 1.0 - target])
    if sparse.issparse(x):
        doubled = sparse.vstack([x, x], format="csr")
    else:
        doubled = np.vstack([x, x])
    return doubled, y, weight


def fit_logistic_pair(x_train, x_eval, y_train, teacher_train, *, c_value: float):
    hard = LogisticRegression(
        C=c_value, penalty="l2", solver="lbfgs", max_iter=2000, random_state=SEED
    ).fit(x_train, y_train)
    mixed = (1.0 - ALPHA) * y_train + ALPHA * teacher_train
    xx, yy, ww = duplicate_fractional_targets(x_train, mixed)
    soft = LogisticRegression(
        C=c_value, penalty="l2", solver="lbfgs", max_iter=2000, random_state=SEED
    ).fit(xx, yy, sample_weight=ww)
    return hard.predict_proba(x_eval)[:, 1], soft.predict_proba(x_eval)[:, 1]


def fit_hashed_pair(sequences_train, sequences_eval, b1_train, b1_eval, y_train, teacher_train):
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(2, 5), n_features=2**16,
        alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32,
    )
    xh_train = vectorizer.transform(sequences_train)
    xh_eval = vectorizer.transform(sequences_eval)
    scaler = StandardScaler().fit(b1_train)
    xb_train = sparse.csr_matrix(scaler.transform(b1_train).astype(np.float32))
    xb_eval = sparse.csr_matrix(scaler.transform(b1_eval).astype(np.float32))
    x_train = sparse.hstack([xh_train, xb_train], format="csr")
    x_eval = sparse.hstack([xh_eval, xb_eval], format="csr")
    settings = dict(
        loss="log_loss", penalty="l2", alpha=1e-4, max_iter=100,
        tol=1e-4, average=True, random_state=SEED, fit_intercept=True,
    )
    hard = SGDClassifier(**settings).fit(x_train, y_train)
    mixed = (1.0 - ALPHA) * y_train + ALPHA * teacher_train
    xx, yy, ww = duplicate_fractional_targets(x_train, mixed)
    soft = SGDClassifier(**settings).fit(xx, yy, sample_weight=ww)
    return (
        hard.predict_proba(x_eval)[:, 1],
        soft.predict_proba(x_eval)[:, 1],
        int(x_train.shape[1]),
    )


def main() -> None:
    started = time.time()
    config = load_config(V1_CONFIG)
    validate_input_hashes(config, include_esm2=False)
    data = load_development(config)
    train = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS))
    evaluation = np.flatnonzero(data.folds == EVALUATION_FOLD)
    if set(data.groups[train]) & set(data.groups[evaluation]):
        raise RuntimeError("Group overlap at the design gate")
    targets = load_teacher_targets(len(data.labels), train)

    design = pd.read_parquet(V1_RUN / "predictions/design.parquet")
    expected_ids = data.record_ids[evaluation]
    if not np.array_equal(design.record_id.to_numpy(dtype=str), expected_ids):
        raise RuntimeError("Frozen fold-0 prediction order mismatch")
    b1_eval_score = design["B1_ORDER_LOGIT"].to_numpy(float)
    teacher_eval_score = design["PROTT5_TEACHER"].to_numpy(float)
    y_eval = data.labels[evaluation]

    output_scores: dict[str, np.ndarray] = {
        "B1_ORDER_LOGIT": b1_eval_score,
        "PROTT5_TEACHER": teacher_eval_score,
    }
    mechanisms: dict[str, object] = {}

    scaler = StandardScaler().fit(data.b1[train])
    b1_train = scaler.transform(data.b1[train])
    b1_eval = scaler.transform(data.b1[evaluation])
    hard, soft = fit_logistic_pair(
        b1_train, b1_eval, data.labels[train], targets[train], c_value=1.0
    )
    output_scores["B1_SOFT_HARD_CONTROL"] = hard
    output_scores["B1_SOFT_STUDENT"] = soft
    mechanisms["B1_SOFT_LOGIT"] = {"feature_count": int(b1_train.shape[1])}

    hard, soft, feature_count = fit_hashed_pair(
        [data.sequences[i] for i in train], [data.sequences[i] for i in evaluation],
        data.b1[train], data.b1[evaluation], data.labels[train], targets[train],
    )
    output_scores["HASHED_HARD_CONTROL"] = hard
    output_scores["HASHED_SOFT_STUDENT"] = soft
    mechanisms["HASHED_2_5MER"] = {"feature_count": feature_count}

    legacy_npz = np.load(LEGACY_ROOT / "_local_soluprot.npz", allow_pickle=True)
    legacy_y = legacy_npz["y"][: len(data.labels)]
    legacy_groups = legacy_npz["cluster"][: len(data.labels)].astype(str)
    legacy_npz.close()
    if not np.array_equal(legacy_y, data.labels) or not np.array_equal(legacy_groups, data.groups):
        raise RuntimeError("Legacy ESM-2 diagnostic alignment failed")
    esm = np.load(LEGACY_ROOT / "_local_soluprot_esm2_full.npy", mmap_mode="r")[: len(data.labels)]
    esm_scaler = StandardScaler().fit(esm[train])
    x_train = np.hstack([b1_train, esm_scaler.transform(esm[train])]).astype(np.float32)
    x_eval = np.hstack([b1_eval, esm_scaler.transform(esm[evaluation])]).astype(np.float32)
    hard, soft = fit_logistic_pair(
        x_train, x_eval, data.labels[train], targets[train], c_value=0.1
    )
    output_scores["LEGACY_ESM35M_HARD_CONTROL"] = hard
    output_scores["LEGACY_ESM35M_SOFT_STUDENT"] = soft
    mechanisms["LEGACY_ESM2_35M"] = {
        "feature_count": int(x_train.shape[1]),
        "diagnostic_only": True,
        "reason": "legacy special-token pooling and 700-residue truncation",
    }

    all_metrics = {name: metrics(y_eval, score) for name, score in output_scores.items()}
    b1_auc = all_metrics["B1_ORDER_LOGIT"]["roc_auc"]
    teacher_auc = all_metrics["PROTT5_TEACHER"]["roc_auc"]
    target_auc = b1_auc + 0.80 * (teacher_auc - b1_auc)
    pairs = {
        "B1_SOFT_LOGIT": ("B1_SOFT_HARD_CONTROL", "B1_SOFT_STUDENT"),
        "HASHED_2_5MER": ("HASHED_HARD_CONTROL", "HASHED_SOFT_STUDENT"),
        "LEGACY_ESM2_35M": ("LEGACY_ESM35M_HARD_CONTROL", "LEGACY_ESM35M_SOFT_STUDENT"),
    }
    gates = {}
    for name, (hard_name, soft_name) in pairs.items():
        soft_auc = all_metrics[soft_name]["roc_auc"]
        hard_auc = all_metrics[hard_name]["roc_auc"]
        gates[name] = {
            "hard_arm": hard_name,
            "soft_arm": soft_name,
            "hard_auc": hard_auc,
            "soft_auc": soft_auc,
            "soft_minus_hard_auc": soft_auc - hard_auc,
            "teacher_gain_retention": (soft_auc - b1_auc) / (teacher_auc - b1_auc),
            "retention_gate_pass": bool(soft_auc >= target_auc),
            "distillation_gain_gate_pass": bool(soft_auc > hard_auc),
        }
        gates[name]["overall_pass"] = bool(
            gates[name]["retention_gate_pass"] and gates[name]["distillation_gain_gate_pass"]
        )

    eligible = [name for name, gate in gates.items() if gate["overall_pass"]]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda name: (
                -gates[name]["soft_auc"],
                all_metrics[gates[name]["soft_arm"]]["log_loss"],
                name,
            ),
        )[0]

    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame({
        "record_id": data.record_ids[evaluation],
        "cluster_id": data.groups[evaluation],
        "development_fold": data.folds[evaluation],
        "label": y_eval,
        **output_scores,
    })
    private_path = PRIVATE_ROOT / "design_predictions.parquet"
    if private_path.exists():
        raise FileExistsError(f"Refusing to overwrite {private_path}")
    prediction_table.to_parquet(private_path, index=False)

    aggregate = {
        "schema": "distillation-rescue-design-dry-run-v1",
        "status": "GO" if selected else "STOP",
        "exploratory_only": True,
        "endpoint": "recombinant soluble-expression status in E. coli",
        "explicit_exclusion": "not direct aqueous solubility of a purified peptide",
        "train_folds": list(TRAIN_FOLDS),
        "evaluation_fold": EVALUATION_FOLD,
        "evaluation_rows": int(len(evaluation)),
        "evaluation_groups": int(len(np.unique(data.groups[evaluation]))),
        "group_overlap": 0,
        "distillation_alpha": ALPHA,
        "b1_auc": b1_auc,
        "teacher_auc": teacher_auc,
        "retention_target": 0.80,
        "required_student_auc": target_auc,
        "metrics": all_metrics,
        "mechanisms": mechanisms,
        "gates": gates,
        "selected_candidate": selected,
        "confirmation_fold_loaded": False,
        "private_prediction_path": str(private_path),
        "runtime_seconds": time.time() - started,
    }
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = PLAN_ROOT / "DESIGN_DRY_RUN_AGGREGATE_20260813.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
