#!/usr/bin/env python3
"""Correct frozen ESM-2 35M branch and no-fit equal ensemble on design fold 0."""

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
PRIVATE_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2"
V1_CONFIG = V1_PLAN / "config/DISTILLATION_CONFIG_20260813.json"
sys.path.insert(0, str(V1_PLAN / "src"))
from teacher_crossfit import load_config, load_development, validate_input_hashes  # noqa: E402


MODEL_ID = "facebook/esm2_t12_35M_UR50D"
TRAIN_FOLDS = (2, 3, 4, 5, 6)
EVALUATION_FOLD = 0
MAX_RESIDUES = 1022
BATCH_SIZE = 16
LOGISTIC_C = 0.1


def bounded_sequence(sequence: str) -> str:
    if len(sequence) <= MAX_RESIDUES:
        return sequence
    left = MAX_RESIDUES // 2
    return sequence[:left] + sequence[-(MAX_RESIDUES - left):]


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def main() -> None:
    started = time.time()
    config = load_config(V1_CONFIG)
    validate_input_hashes(config, include_esm2=False)
    data = load_development(config)
    train = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS))
    evaluation = np.flatnonzero(data.folds == EVALUATION_FOLD)
    selected = np.concatenate([train, evaluation])
    if set(data.groups[train]) & set(data.groups[evaluation]):
        raise RuntimeError("Design group overlap")

    cache = PRIVATE_ROOT / "correct_residue_pool_esm35_design.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as item:
            if not np.array_equal(item["row_indices"], selected):
                raise RuntimeError("Correct ESM cache index mismatch")
            embeddings = item["embeddings"].astype(np.float32)
            extraction_metadata = json.loads(item["metadata_json_utf8"].tobytes().decode("utf-8"))
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).eval().to(device)
        order = selected[np.argsort([len(data.sequences[i]) for i in selected])]
        by_global: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for batch_number, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
                indices = order[start:start + BATCH_SIZE]
                sequences = [bounded_sequence(data.sequences[i]) for i in indices]
                encoded = tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_RESIDUES + 2,
                    return_special_tokens_mask=True,
                )
                tensors = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                ).last_hidden_state
                residue_mask = tensors["attention_mask"] * (1 - tensors["special_tokens_mask"])
                denominator = residue_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
                pooled = ((hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denominator).cpu().numpy()
                for index, vector in zip(indices, pooled):
                    by_global[int(index)] = vector.astype(np.float32)
                if batch_number % 100 == 0:
                    print(f"extracted {min(start + BATCH_SIZE, len(order))}/{len(order)}", flush=True)
        embeddings = np.vstack([by_global[int(index)] for index in selected]).astype(np.float32)
        if embeddings.shape != (len(selected), 480) or not np.isfinite(embeddings).all():
            raise RuntimeError("Correct ESM extraction failed")
        extraction_metadata = {
            "model": MODEL_ID,
            "device": device,
            "pooling": "residue-only arithmetic mean; special tokens excluded",
            "sequence_policy": "full through 1022 residues; first/last concatenation above 1022",
            "row_count": int(len(selected)),
            "dimension": 480,
        }
        PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                row_indices=selected,
                embeddings=embeddings,
                metadata_json_utf8=np.frombuffer(json.dumps(extraction_metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8),
            )
        temporary.replace(cache)

    n_train = len(train)
    train_embedding = embeddings[:n_train]
    evaluation_embedding = embeddings[n_train:]
    b1_scaler = StandardScaler().fit(data.b1[train])
    esm_scaler = StandardScaler().fit(train_embedding)
    x_train = np.hstack([
        b1_scaler.transform(data.b1[train]),
        esm_scaler.transform(train_embedding),
    ]).astype(np.float32)
    x_evaluation = np.hstack([
        b1_scaler.transform(data.b1[evaluation]),
        esm_scaler.transform(evaluation_embedding),
    ]).astype(np.float32)
    head = LogisticRegression(
        C=LOGISTIC_C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=20260813
    ).fit(x_train, data.labels[train])
    frozen_score = head.predict_proba(x_evaluation)[:, 1]

    residual = pd.read_parquet(PRIVATE_ROOT / "esm35_residual_full_predictions.parquet")
    if not np.array_equal(residual.record_id.to_numpy(str), data.record_ids[evaluation]):
        raise RuntimeError("Residual prediction alignment mismatch")
    hard_ensemble = 0.5 * (residual.ESM35_HARD_RESIDUAL.to_numpy(float) + frozen_score)
    distilled_ensemble = 0.5 * (residual.ESM35_DISTILLED_RESIDUAL.to_numpy(float) + frozen_score)
    y = data.labels[evaluation]
    b1 = residual.B1_OFFSET.to_numpy(float)
    teacher = residual.PROTT5_TEACHER.to_numpy(float)
    bundle = {
        "B1_OFFSET": metrics(y, b1),
        "PROTT5_TEACHER": metrics(y, teacher),
        "CORRECT_FROZEN_ESM35_B1": metrics(y, frozen_score),
        "CORRECT_EQUAL_ENSEMBLE_HARD_CONTROL": metrics(y, hard_ensemble),
        "CORRECT_EQUAL_ENSEMBLE_DISTILLED": metrics(y, distilled_ensemble),
    }
    b1_auc = bundle["B1_OFFSET"]["roc_auc"]
    teacher_auc = bundle["PROTT5_TEACHER"]["roc_auc"]
    student_auc = bundle["CORRECT_EQUAL_ENSEMBLE_DISTILLED"]["roc_auc"]
    hard_auc = bundle["CORRECT_EQUAL_ENSEMBLE_HARD_CONTROL"]["roc_auc"]
    required_auc = b1_auc + 0.8 * (teacher_auc - b1_auc)
    aggregate = {
        "schema": "correct-frozen-esm-equal-ensemble-design-v1",
        "status": "DESIGN_PASS" if student_auc >= required_auc and student_auc > hard_auc else "DESIGN_STOP",
        "exploratory_only": True,
        "confirmation_fold_loaded": False,
        "endpoint": "recombinant soluble-expression status in E. coli",
        "explicit_exclusion": "not direct aqueous solubility of a purified peptide",
        "extraction": extraction_metadata,
        "frozen_head": {"logistic_c": LOGISTIC_C, "feature_count": int(x_train.shape[1])},
        "ensemble_rule": "unweighted arithmetic mean of two probabilities; no fitted ensemble weights",
        "metrics": bundle,
        "gates": {
            "required_auc_for_80pct_retention": required_auc,
            "teacher_gain_retention": (student_auc - b1_auc) / (teacher_auc - b1_auc),
            "distilled_minus_hard_ensemble_auc": student_auc - hard_auc,
            "retention_pass": bool(student_auc >= required_auc),
            "distillation_gain_pass": bool(student_auc > hard_auc),
        },
        "runtime_seconds": time.time() - started,
    }
    aggregate["gates"]["overall_pass"] = bool(
        aggregate["gates"]["retention_pass"] and aggregate["gates"]["distillation_gain_pass"]
    )
    private_path = PRIVATE_ROOT / "correct_equal_ensemble_design_predictions.parquet"
    aggregate_path = PLAN_ROOT / "CORRECT_EQUAL_ENSEMBLE_DESIGN_AGGREGATE_20260813.json"
    for path in (private_path, aggregate_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    pd.DataFrame({
        "record_id": data.record_ids[evaluation],
        "cluster_id": data.groups[evaluation],
        "development_fold": data.folds[evaluation],
        "label": y,
        "B1_OFFSET": b1,
        "PROTT5_TEACHER": teacher,
        "CORRECT_FROZEN_ESM35_B1": frozen_score,
        "CORRECT_EQUAL_ENSEMBLE_HARD_CONTROL": hard_ensemble,
        "CORRECT_EQUAL_ENSEMBLE_DISTILLED": distilled_ensemble,
    }).to_parquet(private_path, index=False)
    aggregate["private_prediction_path"] = str(private_path)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
