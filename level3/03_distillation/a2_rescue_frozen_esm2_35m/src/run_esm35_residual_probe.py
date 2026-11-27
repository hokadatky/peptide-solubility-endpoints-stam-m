#!/usr/bin/env python3
"""Bounded ESM-2 35M residual-distillation probe on design folds only.

The model receives a frozen B1 logit as an offset and learns only the residual
needed to approach the cross-fitted ProtT5 teacher.  Smoke mode checks runtime
and numerical stability.  Full mode is a one-epoch exploratory design-fold fit;
it never opens confirmation fold 1.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
V1_PLAN = PROJECT_ROOT / "paper/peptide_solubility_review_distillation_plan_20260813"
V1_RUN = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_run_20260813_v1"
PRIVATE_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2"
V1_CONFIG = V1_PLAN / "config/DISTILLATION_CONFIG_20260813.json"
V4_SRC = PROJECT_ROOT / "paper/peptide_solubility_review/analysis_addon/src"
sys.path[:0] = [str(V1_PLAN / "src"), str(V4_SRC)]

from nested_models import fit_head, predict_head  # noqa: E402
from teacher_crossfit import load_config, load_development, validate_input_hashes  # noqa: E402


MODEL_ID = "facebook/esm2_t12_35M_UR50D"
TRAIN_FOLDS = (2, 3, 4, 5, 6)
EVALUATION_FOLD = 0
SEED = 20260813
MAX_RESIDUES = 1022
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
TOP_LAYERS = 2
ENCODER_LR = 1e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
DISTILLATION_WEIGHT = 0.50


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def metric_bundle(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def teacher_targets(n_rows: int, train: np.ndarray) -> np.ndarray:
    output = np.full(n_rows, np.nan, dtype=float)
    for fold in TRAIN_FOLDS:
        with np.load(
            V1_RUN / f"checkpoints/design/prott5/crossfit_fold_{fold}.npz",
            allow_pickle=False,
        ) as checkpoint:
            indices = checkpoint["row_indices"].astype(int)
            output[indices] = checkpoint["probabilities"].astype(float)
    if not np.isfinite(output[train]).all():
        raise RuntimeError("Teacher targets are incomplete")
    return output


def b1_offsets(data, train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cache = PRIVATE_ROOT / "b1_design_crossfit_offsets.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as item:
            if not np.array_equal(item["train_indices"], train):
                raise RuntimeError("B1 offset cache train-index mismatch")
            if not np.array_equal(item["evaluation_indices"], evaluation):
                raise RuntimeError("B1 offset cache evaluation-index mismatch")
            return item["train_probabilities"].astype(float), item["evaluation_probabilities"].astype(float)
    train_probability = np.full(len(train), np.nan, dtype=float)
    train_position = {int(index): position for position, index in enumerate(train)}
    for heldout in TRAIN_FOLDS:
        fit_indices = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS) & (data.folds != heldout))
        target_indices = np.flatnonzero(data.folds == heldout)
        if set(data.groups[fit_indices]) & set(data.groups[target_indices]):
            raise RuntimeError("B1 crossfit group overlap")
        head = fit_head(data.b1[fit_indices], None, data.labels[fit_indices], "none", c_value=1000.0)
        score = predict_head(head, data.b1[target_indices], None)
        for index, probability in zip(target_indices, score):
            train_probability[train_position[int(index)]] = probability
    if not np.isfinite(train_probability).all():
        raise RuntimeError("B1 crossfit coverage failed")
    head = fit_head(data.b1[train], None, data.labels[train], "none", c_value=1000.0)
    evaluation_probability = predict_head(head, data.b1[evaluation], None)
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_indices=train,
            evaluation_indices=evaluation,
            train_probabilities=train_probability.astype(np.float32),
            evaluation_probabilities=evaluation_probability.astype(np.float32),
        )
    temporary.replace(cache)
    return train_probability, evaluation_probability


def bounded_sequence(sequence: str) -> str:
    sequence = str(sequence)
    if len(sequence) <= MAX_RESIDUES:
        return sequence
    left = MAX_RESIDUES // 2
    return sequence[:left] + sequence[-(MAX_RESIDUES - left):]


class ResidualStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in self.encoder.encoder.layer[-TOP_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.encoder.encoder.emb_layer_norm_after.parameters():
            parameter.requires_grad = True
        self.residual_head = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def forward(self, input_ids, attention_mask, special_tokens_mask, b1_logit):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        residue_mask = attention_mask * (1 - special_tokens_mask)
        denominator = residue_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denominator
        residual = self.residual_head(pooled).squeeze(1)
        return b1_logit + residual, residual


def select_balanced(indices: np.ndarray, labels: np.ndarray, n_rows: int) -> np.ndarray:
    if n_rows >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(SEED)
    half = n_rows // 2
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    chosen = np.concatenate([
        rng.choice(positive, size=half, replace=False),
        rng.choice(negative, size=n_rows - half, replace=False),
    ])
    return np.sort(chosen)


def batches(indices: np.ndarray, sequences: list[str], *, shuffle: bool, batch_size: int):
    order = np.asarray(indices, dtype=int).copy()
    if shuffle:
        rng = np.random.default_rng(SEED)
        rng.shuffle(order)
        blocks = []
        for start in range(0, len(order), 256):
            block = order[start:start + 256]
            blocks.extend(block[np.argsort([len(sequences[i]) for i in block])].tolist())
        order = np.asarray(blocks, dtype=int)
    else:
        order = order[np.argsort([len(sequences[i]) for i in order])]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def run_arm(
    arm: str,
    data,
    tokenizer,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    b1_train_by_global: np.ndarray,
    b1_eval_by_global: np.ndarray,
    teacher_by_global: np.ndarray,
    device: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, torch.Tensor] | None]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = ResidualStudent().to(device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LR},
            {"params": model.residual_head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    bce = torch.nn.BCEWithLogitsLoss()
    smooth = torch.nn.SmoothL1Loss()
    training_loss = []
    model.train()
    for batch_indices in batches(fit_indices, data.sequences, shuffle=True, batch_size=BATCH_SIZE):
        text = [bounded_sequence(data.sequences[i]) for i in batch_indices]
        encoded = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_RESIDUES + 2, return_special_tokens_mask=True,
        )
        tensors = {key: value.to(device) for key, value in encoded.items()}
        b1_tensor = torch.as_tensor(logit(b1_train_by_global[batch_indices]), dtype=torch.float32, device=device)
        label_tensor = torch.as_tensor(data.labels[batch_indices], dtype=torch.float32, device=device)
        final_logit, residual = model(
            tensors["input_ids"], tensors["attention_mask"], tensors["special_tokens_mask"], b1_tensor
        )
        hard_loss = bce(final_logit, label_tensor)
        if arm == "hard":
            loss = hard_loss
            residual_loss = torch.zeros((), device=device)
        else:
            target_residual = np.clip(
                logit(teacher_by_global[batch_indices]) - logit(b1_train_by_global[batch_indices]),
                -4.0, 4.0,
            )
            target_tensor = torch.as_tensor(target_residual, dtype=torch.float32, device=device)
            residual_loss = smooth(residual, target_tensor)
            loss = (1.0 - DISTILLATION_WEIGHT) * hard_loss + DISTILLATION_WEIGHT * residual_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        training_loss.append((float(loss.detach().cpu()), float(hard_loss.detach().cpu()), float(residual_loss.detach().cpu())))

    model.eval()
    probabilities = np.full(len(evaluation_indices), np.nan, dtype=float)
    position = {int(index): pos for pos, index in enumerate(evaluation_indices)}
    with torch.inference_mode():
        for batch_indices in batches(evaluation_indices, data.sequences, shuffle=False, batch_size=EVAL_BATCH_SIZE):
            text = [bounded_sequence(data.sequences[i]) for i in batch_indices]
            encoded = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_RESIDUES + 2, return_special_tokens_mask=True,
            )
            tensors = {key: value.to(device) for key, value in encoded.items()}
            b1_tensor = torch.as_tensor(logit(b1_eval_by_global[batch_indices]), dtype=torch.float32, device=device)
            final_logit, _ = model(
                tensors["input_ids"], tensors["attention_mask"], tensors["special_tokens_mask"], b1_tensor
            )
            score = torch.sigmoid(final_logit).cpu().numpy()
            for index, probability in zip(batch_indices, score):
                probabilities[position[int(index)]] = probability
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{arm} prediction coverage failed")
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    metadata = {
        "arm": arm,
        "trainable_parameters": trainable,
        "total_encoder_parameters": int(sum(p.numel() for p in model.encoder.parameters())),
        "loss_initial": training_loss[0],
        "loss_final": training_loss[-1],
        "batches": len(training_loss),
        "all_losses_finite": bool(np.isfinite(training_loss).all()),
    }
    state = None
    if arm == "distilled":
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model, optimizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return probabilities, metadata, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    started = time.time()
    config = load_config(V1_CONFIG)
    validate_input_hashes(config, include_esm2=False)
    data = load_development(config)
    train_all = np.flatnonzero(np.isin(data.folds, TRAIN_FOLDS))
    evaluation_all = np.flatnonzero(data.folds == EVALUATION_FOLD)
    if set(data.groups[train_all]) & set(data.groups[evaluation_all]):
        raise RuntimeError("Design group overlap")
    target = teacher_targets(len(data.labels), train_all)
    b1_train_vector, b1_eval_vector = b1_offsets(data, train_all, evaluation_all)
    b1_train_global = np.full(len(data.labels), np.nan, dtype=float)
    b1_eval_global = np.full(len(data.labels), np.nan, dtype=float)
    b1_train_global[train_all] = b1_train_vector
    b1_eval_global[evaluation_all] = b1_eval_vector

    frozen_design = pd.read_parquet(V1_RUN / "predictions/design.parquet")
    if not np.array_equal(frozen_design.record_id.to_numpy(str), data.record_ids[evaluation_all]):
        raise RuntimeError("Frozen design prediction alignment failed")
    teacher_eval_global = np.full(len(data.labels), np.nan, dtype=float)
    teacher_eval_global[evaluation_all] = frozen_design.PROTT5_TEACHER.to_numpy(float)
    if args.mode == "smoke":
        fit_indices = select_balanced(train_all, data.labels, 256)
        evaluation_indices = select_balanced(evaluation_all, data.labels, 256)
    else:
        fit_indices = train_all
        evaluation_indices = evaluation_all

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    hard_score, hard_meta, _ = run_arm(
        "hard", data, tokenizer, fit_indices, evaluation_indices,
        b1_train_global, b1_eval_global, target, device,
    )
    distilled_score, distilled_meta, distilled_state = run_arm(
        "distilled", data, tokenizer, fit_indices, evaluation_indices,
        b1_train_global, b1_eval_global, target, device,
    )
    y_eval = data.labels[evaluation_indices]
    b1_score = b1_eval_global[evaluation_indices]
    teacher_score = teacher_eval_global[evaluation_indices]
    result = {
        "schema": "esm35-residual-distillation-probe-v1",
        "mode": args.mode,
        "status": "COMPLETE",
        "exploratory_only": True,
        "confirmation_fold_loaded": False,
        "endpoint": "recombinant soluble-expression status in E. coli",
        "explicit_exclusion": "not direct aqueous solubility of a purified peptide",
        "model": MODEL_ID,
        "device": device,
        "train_folds": list(TRAIN_FOLDS),
        "evaluation_fold": EVALUATION_FOLD,
        "training_rows": int(len(fit_indices)),
        "evaluation_rows": int(len(evaluation_indices)),
        "sequence_policy": "full through 1022 residues; first/last concatenation above 1022",
        "fraction_over_1022_train": float(np.mean(np.asarray([len(data.sequences[i]) for i in fit_indices]) > 1022)),
        "settings": {
            "epochs": 1,
            "top_layers_unfrozen": TOP_LAYERS,
            "encoder_lr": ENCODER_LR,
            "head_lr": HEAD_LR,
            "distillation_weight": DISTILLATION_WEIGHT,
            "batch_size": BATCH_SIZE,
        },
        "metrics": {
            "B1_OFFSET": metric_bundle(y_eval, b1_score),
            "PROTT5_TEACHER": metric_bundle(y_eval, teacher_score),
            "ESM35_HARD_RESIDUAL": metric_bundle(y_eval, hard_score),
            "ESM35_DISTILLED_RESIDUAL": metric_bundle(y_eval, distilled_score),
        },
        "fit_metadata": {"hard": hard_meta, "distilled": distilled_meta},
        "runtime_seconds": time.time() - started,
    }
    b1_auc = result["metrics"]["B1_OFFSET"]["roc_auc"]
    teacher_auc = result["metrics"]["PROTT5_TEACHER"]["roc_auc"]
    student_auc = result["metrics"]["ESM35_DISTILLED_RESIDUAL"]["roc_auc"]
    hard_auc = result["metrics"]["ESM35_HARD_RESIDUAL"]["roc_auc"]
    required_auc = b1_auc + 0.80 * (teacher_auc - b1_auc)
    result["gates"] = {
        "required_auc_for_80pct_retention": required_auc,
        "teacher_gain_retention": (student_auc - b1_auc) / (teacher_auc - b1_auc),
        "distilled_minus_hard_auc": student_auc - hard_auc,
        "retention_pass": bool(student_auc >= required_auc),
        "distillation_gain_pass": bool(student_auc > hard_auc),
    }
    result["gates"]["overall_pass"] = bool(
        result["gates"]["retention_pass"] and result["gates"]["distillation_gain_pass"]
    )

    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = PRIVATE_ROOT / f"esm35_residual_{args.mode}_predictions.parquet"
    aggregate_path = PLAN_ROOT / f"ESM35_RESIDUAL_{args.mode.upper()}_AGGREGATE_20260813.json"
    for path in (prediction_path, aggregate_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    pd.DataFrame({
        "record_id": data.record_ids[evaluation_indices],
        "cluster_id": data.groups[evaluation_indices],
        "development_fold": data.folds[evaluation_indices],
        "label": y_eval,
        "B1_OFFSET": b1_score,
        "PROTT5_TEACHER": teacher_score,
        "ESM35_HARD_RESIDUAL": hard_score,
        "ESM35_DISTILLED_RESIDUAL": distilled_score,
    }).to_parquet(prediction_path, index=False)
    result["private_prediction_path"] = str(prediction_path)
    if args.mode == "full" and result["gates"]["overall_pass"] and distilled_state is not None:
        artifact = PRIVATE_ROOT / "ESM35_DISTILLED_RESIDUAL_design_pass.pt"
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        torch.save({"state_dict": distilled_state, "settings": result["settings"]}, artifact)
        result["private_artifact_path"] = str(artifact)
        result["private_artifact_bytes"] = artifact.stat().st_size
    aggregate_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
