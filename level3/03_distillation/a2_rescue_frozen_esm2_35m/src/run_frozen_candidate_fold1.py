#!/usr/bin/env python3
"""One post-selection fold-1 robustness check of the frozen rescue candidate.

Fold 1 was opened in V1 and is therefore not an untouched confirmation here.
The script makes no choices from fold-1 outcomes and permits no rerun settings.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
V1_PLAN = PROJECT_ROOT / "paper/peptide_solubility_review_distillation_plan_20260813"
V1_RUN = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_run_20260813_v1"
PRIVATE_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2"
V1_CONFIG = V1_PLAN / "config/DISTILLATION_CONFIG_20260813.json"
V4_SRC = PROJECT_ROOT / "paper/peptide_solubility_review/analysis_addon/src"
sys.path[:0] = [str(PLAN_ROOT / "src"), str(V1_PLAN / "src"), str(V4_SRC)]

from nested_models import fit_head, predict_head  # noqa: E402
from teacher_crossfit import load_config, load_development, validate_input_hashes  # noqa: E402
from run_esm35_residual_probe import (  # noqa: E402
    MODEL_ID,
    MAX_RESIDUES,
    run_arm,
    bounded_sequence,
)


TRAIN_FOLDS = (0, 2, 3, 4, 5, 6)
EVALUATION_FOLD = 1
LOGISTIC_C = 0.1
EXTRACTION_BATCH_SIZE = 16


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def load_teacher_targets(n_rows: int, train: np.ndarray) -> np.ndarray:
    output = np.full(n_rows, np.nan, dtype=float)
    for fold in TRAIN_FOLDS:
        with np.load(
            V1_RUN / f"checkpoints/confirmation/prott5/crossfit_fold_{fold}.npz",
            allow_pickle=False,
        ) as checkpoint:
            output[checkpoint["row_indices"].astype(int)] = checkpoint["probabilities"].astype(float)
    if not np.isfinite(output[train]).all():
        raise RuntimeError("Confirmation cross-fitted teacher targets are incomplete")
    return output


def confirmation_b1_offsets(data, train: np.ndarray, evaluation: np.ndarray):
    cache = PRIVATE_ROOT / "b1_postselection_fold1_offsets.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as item:
            if not np.array_equal(item["train_indices"], train) or not np.array_equal(item["evaluation_indices"], evaluation):
                raise RuntimeError("B1 fold-1 cache index mismatch")
            return item["train_probabilities"].astype(float), item["evaluation_probabilities"].astype(float)
    train_probability_global = np.full(len(data.labels), np.nan, dtype=float)
    for heldout in TRAIN_FOLDS:
        fit_indices = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS) & (data.folds != heldout))
        target_indices = np.flatnonzero(data.folds == heldout)
        if set(data.groups[fit_indices]) & set(data.groups[target_indices]):
            raise RuntimeError("B1 crossfit group overlap")
        head = fit_head(data.b1[fit_indices], None, data.labels[fit_indices], "none", c_value=1000.0)
        train_probability_global[target_indices] = predict_head(head, data.b1[target_indices], None)
    if not np.isfinite(train_probability_global[train]).all():
        raise RuntimeError("B1 crossfit coverage failed")
    frozen = pd.read_parquet(V1_RUN / "predictions/confirmation.parquet")
    if not np.array_equal(frozen.record_id.to_numpy(str), data.record_ids[evaluation]):
        raise RuntimeError("V1 confirmation order mismatch")
    evaluation_probability = frozen.B1_ORDER_LOGIT.to_numpy(float)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_indices=train,
            evaluation_indices=evaluation,
            train_probabilities=train_probability_global[train].astype(np.float32),
            evaluation_probabilities=evaluation_probability.astype(np.float32),
        )
    temporary.replace(cache)
    return train_probability_global[train], evaluation_probability


def extract_fold1_correct_esm(data, evaluation: np.ndarray, device: str) -> np.ndarray:
    cache = PRIVATE_ROOT / "correct_residue_pool_esm35_fold1.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as item:
            if not np.array_equal(item["row_indices"], evaluation):
                raise RuntimeError("Fold-1 ESM cache index mismatch")
            return item["embeddings"].astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).eval().to(device)
    order = evaluation[np.argsort([len(data.sequences[i]) for i in evaluation])]
    by_global = {}
    with torch.inference_mode():
        for start in range(0, len(order), EXTRACTION_BATCH_SIZE):
            indices = order[start:start + EXTRACTION_BATCH_SIZE]
            encoded = tokenizer(
                [bounded_sequence(data.sequences[i]) for i in indices],
                return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_RESIDUES + 2, return_special_tokens_mask=True,
            )
            tensors = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(
                input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]
            ).last_hidden_state
            residue_mask = tensors["attention_mask"] * (1 - tensors["special_tokens_mask"])
            denominator = residue_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
            pooled = ((hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denominator).cpu().numpy()
            for index, vector in zip(indices, pooled):
                by_global[int(index)] = vector.astype(np.float32)
    result = np.vstack([by_global[int(index)] for index in evaluation]).astype(np.float32)
    if result.shape != (len(evaluation), 480) or not np.isfinite(result).all():
        raise RuntimeError("Fold-1 correct ESM extraction failed")
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row_indices=evaluation, embeddings=result)
    temporary.replace(cache)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return result


def load_design_correct_esm(train: np.ndarray) -> np.ndarray:
    cache = PRIVATE_ROOT / "correct_residue_pool_esm35_design.npz"
    with np.load(cache, allow_pickle=False) as item:
        indices = item["row_indices"].astype(int)
        embeddings = item["embeddings"].astype(np.float32)
    lookup = {int(index): vector for index, vector in zip(indices, embeddings)}
    if any(int(index) not in lookup for index in train):
        raise RuntimeError("Design correct ESM cache does not cover fold-1 training set")
    return np.vstack([lookup[int(index)] for index in train]).astype(np.float32)


def main() -> None:
    started = time.time()
    config = load_config(V1_CONFIG)
    validate_input_hashes(config, include_esm2=False)
    data = load_development(config)
    train = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS))
    evaluation = np.flatnonzero(data.folds == EVALUATION_FOLD)
    if set(data.groups[train]) & set(data.groups[evaluation]):
        raise RuntimeError("Fold-1 group overlap")
    teacher = load_teacher_targets(len(data.labels), train)
    print("teacher targets: PASS", flush=True)
    b1_train, b1_evaluation = confirmation_b1_offsets(data, train, evaluation)
    b1_train_global = np.full(len(data.labels), np.nan, dtype=float)
    b1_evaluation_global = np.full(len(data.labels), np.nan, dtype=float)
    b1_train_global[train] = b1_train
    b1_evaluation_global[evaluation] = b1_evaluation
    print("B1 crossfit offsets: PASS", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    print(f"residual hard fit start: device={device}", flush=True)
    hard_score, hard_meta, _ = run_arm(
        "hard", data, tokenizer, train, evaluation,
        b1_train_global, b1_evaluation_global, teacher, device,
    )
    print("residual hard fit/evaluation: PASS", flush=True)
    distilled_score, distilled_meta, distilled_state = run_arm(
        "distilled", data, tokenizer, train, evaluation,
        b1_train_global, b1_evaluation_global, teacher, device,
    )
    print("residual distilled fit/evaluation: PASS", flush=True)

    correct_train = load_design_correct_esm(train)
    correct_evaluation = extract_fold1_correct_esm(data, evaluation, device)
    b1_scaler = StandardScaler().fit(data.b1[train])
    esm_scaler = StandardScaler().fit(correct_train)
    x_train = np.hstack([
        b1_scaler.transform(data.b1[train]), esm_scaler.transform(correct_train)
    ]).astype(np.float32)
    x_evaluation = np.hstack([
        b1_scaler.transform(data.b1[evaluation]), esm_scaler.transform(correct_evaluation)
    ]).astype(np.float32)
    frozen_head = LogisticRegression(
        C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=20260813
    ).fit(x_train, data.labels[train])
    frozen_score = frozen_head.predict_proba(x_evaluation)[:, 1]
    hard_ensemble = 0.5 * (hard_score + frozen_score)
    distilled_ensemble = 0.5 * (distilled_score + frozen_score)

    frozen_v1 = pd.read_parquet(V1_RUN / "predictions/confirmation.parquet")
    if not np.array_equal(frozen_v1.record_id.to_numpy(str), data.record_ids[evaluation]):
        raise RuntimeError("V1 fold-1 alignment mismatch")
    teacher_evaluation = frozen_v1.PROTT5_TEACHER.to_numpy(float)
    y = data.labels[evaluation]
    bundle = {
        "B1_OFFSET": metrics(y, b1_evaluation),
        "PROTT5_TEACHER": metrics(y, teacher_evaluation),
        "ESM35_HARD_RESIDUAL": metrics(y, hard_score),
        "ESM35_DISTILLED_RESIDUAL": metrics(y, distilled_score),
        "CORRECT_FROZEN_ESM35_B1": metrics(y, frozen_score),
        "FROZEN_EQUAL_ENSEMBLE_HARD_CONTROL": metrics(y, hard_ensemble),
        "FROZEN_EQUAL_ENSEMBLE_DISTILLED": metrics(y, distilled_ensemble),
    }
    b1_auc = bundle["B1_OFFSET"]["roc_auc"]
    teacher_auc = bundle["PROTT5_TEACHER"]["roc_auc"]
    student_auc = bundle["FROZEN_EQUAL_ENSEMBLE_DISTILLED"]["roc_auc"]
    hard_auc = bundle["FROZEN_EQUAL_ENSEMBLE_HARD_CONTROL"]["roc_auc"]
    required_auc = b1_auc + 0.8 * (teacher_auc - b1_auc)
    aggregate = {
        "schema": "frozen-rescue-postselection-fold1-v1",
        "status": "ROBUSTNESS_PASS" if student_auc >= required_auc and student_auc > hard_auc else "ROBUSTNESS_STOP",
        "evaluation_role": "post-selection robustness check; fold 1 was already opened in V1 and is not an untouched confirmation",
        "endpoint": "recombinant soluble-expression status in E. coli",
        "explicit_exclusion": "not direct aqueous solubility of a purified peptide",
        "train_folds": list(TRAIN_FOLDS),
        "evaluation_fold": EVALUATION_FOLD,
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_groups": int(len(np.unique(data.groups[evaluation]))),
        "group_overlap": 0,
        "device": device,
        "frozen_design_settings_changed": False,
        "metrics": bundle,
        "fit_metadata": {"hard": hard_meta, "distilled": distilled_meta},
        "gates": {
            "required_auc_for_80pct_retention": required_auc,
            "teacher_gain_retention": (student_auc - b1_auc) / (teacher_auc - b1_auc),
            "distilled_minus_hard_ensemble_auc": student_auc - hard_auc,
            "retention_pass": bool(student_auc >= required_auc),
            "distillation_gain_pass": bool(student_auc > hard_auc),
        },
        "runtime_seconds": time.time() - started,
        "claim_ceiling": "Retrospective robustness only; not independent confirmation, external validation, or app-promotion evidence.",
    }
    aggregate["gates"]["overall_pass"] = bool(
        aggregate["gates"]["retention_pass"] and aggregate["gates"]["distillation_gain_pass"]
    )
    private_predictions = PRIVATE_ROOT / "frozen_candidate_postselection_fold1_predictions.parquet"
    aggregate_path = PLAN_ROOT / "FROZEN_CANDIDATE_POSTSELECTION_FOLD1_AGGREGATE_20260813.json"
    for path in (private_predictions, aggregate_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    pd.DataFrame({
        "record_id": data.record_ids[evaluation],
        "cluster_id": data.groups[evaluation],
        "development_fold": data.folds[evaluation],
        "label": y,
        "B1_OFFSET": b1_evaluation,
        "PROTT5_TEACHER": teacher_evaluation,
        "ESM35_HARD_RESIDUAL": hard_score,
        "ESM35_DISTILLED_RESIDUAL": distilled_score,
        "CORRECT_FROZEN_ESM35_B1": frozen_score,
        "FROZEN_EQUAL_ENSEMBLE_HARD_CONTROL": hard_ensemble,
        "FROZEN_EQUAL_ENSEMBLE_DISTILLED": distilled_ensemble,
    }).to_parquet(private_predictions, index=False)
    aggregate["private_prediction_path"] = str(private_predictions)
    if aggregate["gates"]["overall_pass"] and distilled_state is not None:
        artifact = PRIVATE_ROOT / "FROZEN_RESCUE_ESM35_DISTILLED_fold1_robustness.pt"
        torch.save({"state_dict": distilled_state, "design_settings": aggregate["fit_metadata"]["distilled"]}, artifact)
        aggregate["private_distilled_artifact_path"] = str(artifact)
        aggregate["private_distilled_artifact_bytes"] = artifact.stat().st_size
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
