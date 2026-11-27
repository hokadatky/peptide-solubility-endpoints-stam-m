#!/usr/bin/env python3
"""Run one bounded stage of the frozen TriPepLite distillation protocol.

This runner deliberately has no hyperparameter arguments.  Scientific choices
come only from the hashed JSON contract.  Row-level outputs are written only to
the private run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


sys.dont_write_bytecode = True


PLAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAN_ROOT.parents[1]
V4_SRC = (
    PROJECT_ROOT / "paper" / "peptide_solubility_review" / "analysis_addon" / "src"
)
sys.path.insert(0, str(PLAN_ROOT / "src"))
sys.path.insert(0, str(V4_SRC))

from clustered_stats import metric_bundle, paired_cluster_bootstrap  # noqa: E402
from nested_models import fit_head, predict_head  # noqa: E402
from teacher_crossfit import (  # noqa: E402
    DevelopmentData,
    crossfit_plan_manifest,
    fixed_teacher_crossfit,
    fixed_teacher_evaluation,
    load_config,
    load_development,
    load_esm2,
    load_prott5,
    resolve_project_path,
    sha256_file,
    stable_hash,
    teacher_embedding,
    validate_esm2_final,
    validate_input_hashes,
)
from tripeplite import (  # noqa: E402
    export_numpy_bundle,
    load_numpy_bundle,
    predict_numpy_bundle,
    train_fixed_student,
)


def atomic_json(payload: object, path: Path, *, refuse_conflict: bool = True) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == encoded:
            return
        if refuse_conflict:
            raise FileExistsError(f"Refusing to overwrite a different JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def environment_manifest() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "primary_student_device": "cpu",
    }


def stage_contract(config: dict[str, object], stage: str) -> tuple[list[int], int]:
    if stage == "design":
        return (
            [int(value) for value in config["data_contract"]["design_train_folds"]],
            int(config["data_contract"]["design_gate_fold"]),
        )
    if stage == "confirmation":
        return (
            [
                int(value)
                for value in config["data_contract"]["confirmation_train_folds"]
            ],
            int(config["data_contract"]["confirmation_fold"]),
        )
    raise ValueError(f"Unsupported stage: {stage}")


def fit_or_load_baseline(
    data: DevelopmentData,
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    config: dict[str, object],
    run_root: Path,
    stage: str,
) -> np.ndarray:
    path = run_root / "checkpoints" / stage / "b1" / "evaluation.npz"
    contract = stable_hash(
        {
            "run_id": config["run_id"],
            "stage": stage,
            "arm": "B1_ORDER_LOGIT",
            "train_folds": sorted(np.unique(data.folds[train]).astype(int).tolist()),
            "evaluation_fold": int(np.unique(data.folds[evaluation])[0]),
            "evaluation_index_hash": stable_hash(evaluation.astype(int).tolist()),
            "c": float(config["teacher"]["logistic_c"]),
        }
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as checkpoint:
            if str(checkpoint["contract_hash"][0]) != contract:
                raise ValueError(f"B1 checkpoint contract mismatch: {path}")
            if not np.array_equal(checkpoint["row_indices"], evaluation):
                raise ValueError(f"B1 checkpoint row mismatch: {path}")
            return checkpoint["probabilities"].astype(np.float64)
    head = fit_head(
        data.b1[train],
        None,
        data.labels[train],
        "none",
        c_value=float(config["teacher"]["logistic_c"]),
    )
    score = predict_head(head, data.b1[evaluation], None)
    score = score.astype(np.float32).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            contract_hash=np.asarray([contract]),
            row_indices=evaluation.astype(np.int64),
            probabilities=score.astype(np.float32),
        )
    temporary.replace(path)
    return score.astype(np.float32).astype(np.float64)


def student_contract_hash(
    config: dict[str, object],
    *,
    stage: str,
    arm: str,
    seed: int,
    train: np.ndarray,
    teacher_targets: np.ndarray | None,
) -> str:
    if teacher_targets is None:
        teacher_target_hash = "hard_only_no_teacher_targets"
    else:
        selected_targets = np.asarray(teacher_targets, dtype=np.float64)[train]
        if selected_targets.shape != (len(train),) or not np.isfinite(
            selected_targets
        ).all():
            raise ValueError("Student teacher targets are absent or non-finite.")
        canonical_targets = np.ascontiguousarray(selected_targets.astype("<f4"))
        teacher_target_hash = hashlib.sha256(canonical_targets.tobytes()).hexdigest()
    return stable_hash(
        {
            "run_id": config["run_id"],
            "stage": stage,
            "arm": arm,
            "seed": int(seed),
            "train_index_hash": stable_hash(train.astype(int).tolist()),
            "teacher_target_float32_sha256": teacher_target_hash,
            "student": config["student"],
            "training": config["training"],
            "development_sha256": config["input_sha256"]["development"],
            "features_sha256": config["input_sha256"]["transparent_features"],
        }
    )


def fit_or_load_student(
    data: DevelopmentData,
    train: np.ndarray,
    evaluation: np.ndarray,
    teacher_targets: np.ndarray | None,
    *,
    alpha: float,
    seed: int,
    arm: str,
    config: dict[str, object],
    run_root: Path,
    stage: str,
) -> tuple[np.ndarray, dict[str, object]]:
    artifact = run_root / "models" / stage / f"{arm}_seed{int(seed)}.npz"
    history_path = run_root / "models" / stage / f"{arm}_seed{int(seed)}_history.json"
    contract = student_contract_hash(
        config,
        stage=stage,
        arm=arm,
        seed=seed,
        train=train,
        teacher_targets=teacher_targets,
    )
    if artifact.exists():
        metadata, _ = load_numpy_bundle(artifact)
        if metadata.get("training_contract_hash") != contract:
            raise ValueError(f"Student artifact contract mismatch: {artifact}")
    else:
        fit = train_fixed_student(
            [data.sequences[int(index)] for index in train],
            data.b0[train],
            data.labels[train],
            None if teacher_targets is None else teacher_targets[train],
            seed=int(seed),
            epochs=int(config["training"]["epochs"]),
            batch_size=int(config["training"]["batch_size"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            alpha=float(alpha),
            temperature=float(config["training"]["temperature"]),
            terminal_residues=int(config["student"]["terminal_residues"]),
            device=str(config["student"]["primary_device"]),
            torch_threads=1,
        )
        metadata = export_numpy_bundle(
            fit,
            artifact,
            descriptor_names=data.b0_names,
            endpoint=str(config["endpoint"]["machine_name"]),
            terminal_residues=int(config["student"]["terminal_residues"]),
            extra_metadata={
                "run_id": config["run_id"],
                "stage": stage,
                "arm": arm,
                "training_contract_hash": contract,
                "training_rows": int(len(train)),
                "training_groups": int(len(np.unique(data.groups[train]))),
                "endpoint_boundary": config["endpoint"]["explicit_exclusion"],
            },
        )
        atomic_json(
            {
                "run_id": config["run_id"],
                "stage": stage,
                "arm": arm,
                "seed": int(seed),
                "training_contract_hash": contract,
                "epochs": len(fit.history),
                "initial_loss": fit.history[0],
                "final_loss": fit.history[-1],
                "all_losses_finite": bool(
                    np.isfinite([row["total_loss"] for row in fit.history]).all()
                ),
                "artifact_sha256": metadata["artifact_sha256"],
                "artifact_bytes": metadata["artifact_bytes"],
                "contains_row_level_data": False,
            },
            history_path,
        )
    score = predict_numpy_bundle(
        artifact,
        [data.sequences[int(index)] for index in evaluation],
        data.b0[evaluation],
    )
    metadata, _ = load_numpy_bundle(artifact)
    metadata = dict(metadata)
    metadata["artifact_sha256"] = sha256_file(artifact)
    metadata["artifact_bytes"] = artifact.stat().st_size
    if artifact.stat().st_size > int(config["student"]["artifact_size_limit_bytes"]):
        raise RuntimeError("Student artifact exceeds the frozen size gate.")
    return score, metadata


def write_private_predictions(
    data: DevelopmentData,
    evaluation: np.ndarray,
    scores: dict[str, np.ndarray],
    path: Path,
) -> None:
    table = pd.DataFrame(
        {
            "record_id": data.record_ids[evaluation],
            "cluster_id": data.groups[evaluation],
            "development_fold": data.folds[evaluation],
            "label": data.labels[evaluation],
        }
    )
    for name, score in scores.items():
        if np.asarray(score).shape != (len(evaluation),):
            raise ValueError(f"Prediction shape mismatch for {name}")
        table[name] = np.asarray(score, dtype=np.float64)
    if path.exists():
        existing = pd.read_parquet(path)
        if list(existing.columns) != list(table.columns) or len(existing) != len(table):
            raise FileExistsError(f"Refusing incompatible existing prediction file: {path}")
        numeric = [column for column in table if column not in {"record_id", "cluster_id"}]
        if not (
            np.array_equal(existing.record_id.to_numpy(), table.record_id.to_numpy())
            and np.array_equal(existing.cluster_id.to_numpy(), table.cluster_id.to_numpy())
            and np.allclose(existing[numeric], table[numeric], rtol=0.0, atol=1e-7)
        ):
            raise FileExistsError(f"Refusing different existing prediction file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def aggregate_metrics(
    labels: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    return {name: metric_bundle(labels, score) for name, score in scores.items()}


def paired_student_bootstrap(
    labels: np.ndarray,
    hard_score: np.ndarray,
    distilled_score: np.ndarray,
    groups: np.ndarray,
    config: dict[str, object],
) -> dict[str, object]:
    """Apply the fixed paired cluster bootstrap to distilled minus hard-only.

    Keeping this contrast in a small helper allows its direction, resampling
    unit and metric set to be contract-tested without starting a scientific
    model fit.
    """
    return {
        metric: paired_cluster_bootstrap(
            labels,
            hard_score,
            distilled_score,
            groups,
            metric=metric,
            replicates=int(config["confirmation"]["bootstrap_replicates"]),
            seed=int(config["confirmation"]["bootstrap_seed"]),
        )
        for metric in ("roc_auc", "auprc", "brier", "log_loss")
    }


def hybrid_gate(
    metrics: dict[str, dict[str, float]], config: dict[str, object]
) -> dict[str, object]:
    gate = config["hybrid_selection_gate"]
    teacher_loss_improvement = (
        metrics["PROTT5_TEACHER"]["log_loss"]
        - metrics["PROTT5_ESM2_TEACHER"]["log_loss"]
    )
    student_loss_improvement = (
        metrics["PROTT5_DISTILLED_TRIPEPLITE"]["log_loss"]
        - metrics["PROTT5_ESM2_DISTILLED_TRIPEPLITE"]["log_loss"]
    )
    teacher_auc_decrement = (
        metrics["PROTT5_TEACHER"]["roc_auc"]
        - metrics["PROTT5_ESM2_TEACHER"]["roc_auc"]
    )
    student_auc_decrement = (
        metrics["PROTT5_DISTILLED_TRIPEPLITE"]["roc_auc"]
        - metrics["PROTT5_ESM2_DISTILLED_TRIPEPLITE"]["roc_auc"]
    )
    checks = {
        "teacher_log_loss_improvement_pass": bool(
            teacher_loss_improvement
            >= float(gate["minimum_teacher_log_loss_improvement"])
        ),
        "student_log_loss_improvement_pass": bool(
            student_loss_improvement
            >= float(gate["minimum_student_log_loss_improvement"])
        ),
        "teacher_auc_guard_pass": bool(
            teacher_auc_decrement
            <= float(gate["maximum_allowed_teacher_auc_decrement"])
        ),
        "student_auc_guard_pass": bool(
            student_auc_decrement
            <= float(gate["maximum_allowed_student_auc_decrement"])
        ),
    }
    selected = "hybrid" if all(checks.values()) else "prott5"
    return {
        "status": "FROZEN_SELECTION_COMPLETE",
        "selection_dataset": "development_fold_0_only",
        "selected_teacher": selected,
        "fallback_applied": selected == "prott5",
        "observed": {
            "teacher_log_loss_improvement": teacher_loss_improvement,
            "student_log_loss_improvement": student_loss_improvement,
            "teacher_auc_decrement": teacher_auc_decrement,
            "student_auc_decrement": student_auc_decrement,
        },
        "thresholds": gate,
        "checks": checks,
        "confirmation_fold_was_not_used": True,
    }


def run_preflight(config: dict[str, object], run_root: Path) -> None:
    validate_input_hashes(config, include_esm2=False)
    data = load_development(config)
    prott5 = load_prott5(config)
    esm_status: dict[str, object]
    try:
        metadata = validate_esm2_final(config)
        esm_status = {
            "complete": True,
            "array_sha256": metadata["output_sha256"],
            "n_records": metadata["n_records"],
        }
    except FileNotFoundError as error:
        shard_dir = resolve_project_path(config, "esm2").parent / "esm2_shards"
        shards = sorted(shard_dir.glob("*.npz"))
        esm_status = {
            "complete": False,
            "existing_shards": len(shards),
            "expected_shards": 1456,
            "missing_shards": 1456 - len(shards),
            "reason": str(error),
        }
    payload = {
        "status": "PASS_WITH_ESM2_PENDING" if not esm_status["complete"] else "PASS",
        "run_id": config["run_id"],
        "environment": environment_manifest(),
        "development": {
            "rows": len(data.labels),
            "groups": len(np.unique(data.groups)),
            "folds": sorted(np.unique(data.folds).astype(int).tolist()),
            "b0_dimension": data.b0.shape[1],
            "b1_dimension": data.b1.shape[1],
        },
        "prott5_shape": list(prott5.shape),
        "esm2": esm_status,
        "design_crossfit": crossfit_plan_manifest(
            data, config["data_contract"]["design_train_folds"]
        ),
        "confirmation_crossfit": crossfit_plan_manifest(
            data, config["data_contract"]["confirmation_train_folds"]
        ),
        "contains_row_level_data": False,
    }
    status_token = str(payload["status"]).lower()
    atomic_json(payload, run_root / "manifests" / f"preflight_{status_token}.json")
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_stage(config: dict[str, object], run_root: Path, stage: str) -> None:
    started = time.perf_counter()
    validate_input_hashes(config, include_esm2=(stage == "design"))
    data = load_development(config)
    prott5 = load_prott5(config)
    train_folds, evaluation_fold = stage_contract(config, stage)
    train = np.flatnonzero(np.isin(data.folds, train_folds))
    evaluation = np.flatnonzero(data.folds == evaluation_fold)

    teacher_names = ["prott5", "hybrid"] if stage == "design" else []
    selection: dict[str, object] | None = None
    if stage == "confirmation":
        selection_path = run_root / "aggregate" / "hybrid_selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError("Design-stage hybrid selection is absent.")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        teacher_names = [str(selection["selected_teacher"])]

    esm2_metadata = validate_esm2_final(config) if "hybrid" in teacher_names else None
    esm2 = load_esm2(config) if "hybrid" in teacher_names else None
    embeddings = {
        name: teacher_embedding(config, name, prott5, esm2) for name in teacher_names
    }
    embedding_source_hashes = {
        "prott5": str(config["input_sha256"]["prott5"]),
        "hybrid": stable_hash(
            {
                "prott5": config["input_sha256"]["prott5"],
                "esm2": None if esm2_metadata is None else esm2_metadata["output_sha256"],
            }
        ),
    }
    teacher_targets: dict[str, np.ndarray] = {}
    teacher_evaluation: dict[str, np.ndarray] = {}
    for teacher_name in teacher_names:
        teacher_targets[teacher_name] = fixed_teacher_crossfit(
            data,
            embeddings[teacher_name],
            train_folds,
            config=config,
            run_root=run_root,
            stage=stage,
            teacher_name=teacher_name,
            embedding_source_hash=embedding_source_hashes[teacher_name],
        )
        teacher_indices, teacher_score = fixed_teacher_evaluation(
            data,
            embeddings[teacher_name],
            train_folds,
            evaluation_fold,
            config=config,
            run_root=run_root,
            stage=stage,
            teacher_name=teacher_name,
            embedding_source_hash=embedding_source_hashes[teacher_name],
        )
        if not np.array_equal(teacher_indices, evaluation):
            raise RuntimeError("Teacher/student evaluation rows differ.")
        teacher_evaluation[teacher_name] = teacher_score

    score: dict[str, np.ndarray] = {
        "B1_ORDER_LOGIT": fit_or_load_baseline(
            data,
            train,
            evaluation,
            config=config,
            run_root=run_root,
            stage=stage,
        )
    }
    artifact_metadata: dict[str, dict[str, object]] = {}
    primary_seed = int(config["training"]["primary_seed"])
    hard_score, hard_metadata = fit_or_load_student(
        data,
        train,
        evaluation,
        None,
        alpha=float(config["training"]["hard_only_alpha"]),
        seed=primary_seed,
        arm="TRIPEPLITE_HARD",
        config=config,
        run_root=run_root,
        stage=stage,
    )
    score["TRIPEPLITE_HARD"] = hard_score
    artifact_metadata["TRIPEPLITE_HARD"] = hard_metadata

    for teacher_name in teacher_names:
        teacher_arm = (
            "PROTT5_TEACHER" if teacher_name == "prott5" else "PROTT5_ESM2_TEACHER"
        )
        student_arm = (
            "PROTT5_DISTILLED_TRIPEPLITE"
            if teacher_name == "prott5"
            else "PROTT5_ESM2_DISTILLED_TRIPEPLITE"
        )
        score[teacher_arm] = teacher_evaluation[teacher_name]
        student_score, metadata = fit_or_load_student(
            data,
            train,
            evaluation,
            teacher_targets[teacher_name],
            alpha=float(config["training"]["distillation_alpha"]),
            seed=primary_seed,
            arm=student_arm,
            config=config,
            run_root=run_root,
            stage=stage,
        )
        score[student_arm] = student_score
        artifact_metadata[student_arm] = metadata

    sensitivity: dict[str, object] = {}
    if stage == "confirmation":
        selected_teacher = teacher_names[0]
        selected_student_arm = (
            "PROTT5_DISTILLED_TRIPEPLITE"
            if selected_teacher == "prott5"
            else "PROTT5_ESM2_DISTILLED_TRIPEPLITE"
        )
        for seed in config["training"]["sensitivity_seeds"]:
            seed = int(seed)
            hard_seed_score, _ = fit_or_load_student(
                data,
                train,
                evaluation,
                None,
                alpha=float(config["training"]["hard_only_alpha"]),
                seed=seed,
                arm="TRIPEPLITE_HARD",
                config=config,
                run_root=run_root,
                stage=stage,
            )
            distill_seed_score, _ = fit_or_load_student(
                data,
                train,
                evaluation,
                teacher_targets[selected_teacher],
                alpha=float(config["training"]["distillation_alpha"]),
                seed=seed,
                arm=selected_student_arm,
                config=config,
                run_root=run_root,
                stage=stage,
            )
            score[f"TRIPEPLITE_HARD_seed{seed}"] = hard_seed_score
            score[f"{selected_student_arm}_seed{seed}"] = distill_seed_score
            sensitivity[str(seed)] = {
                "hard": metric_bundle(data.labels[evaluation], hard_seed_score),
                "distilled": metric_bundle(data.labels[evaluation], distill_seed_score),
            }

    write_private_predictions(
        data, evaluation, score, run_root / "predictions" / f"{stage}.parquet"
    )
    metrics = aggregate_metrics(data.labels[evaluation], score)
    aggregate: dict[str, object] = {
        "status": "PASS",
        "run_id": config["run_id"],
        "stage": stage,
        "analysis_scope": config["analysis_scope"],
        "endpoint": config["endpoint"],
        "train_folds": train_folds,
        "evaluation_fold": evaluation_fold,
        "n_train_rows": int(len(train)),
        "n_train_groups": int(len(np.unique(data.groups[train]))),
        "n_evaluation_rows": int(len(evaluation)),
        "n_evaluation_groups": int(len(np.unique(data.groups[evaluation]))),
        "metrics": metrics,
        "artifact_metadata": artifact_metadata,
        "sensitivity": sensitivity,
        "environment": environment_manifest(),
        "privacy": {
            "contains_record_ids": False,
            "contains_sequences": False,
            "contains_labels": False,
            "contains_row_level_predictions": False,
            "private_prediction_path": str(
                run_root / "predictions" / f"{stage}.parquet"
            ),
        },
    }
    if stage == "design":
        selection_payload = hybrid_gate(metrics, config)
        aggregate["hybrid_selection"] = selection_payload
        atomic_json(
            selection_payload, run_root / "aggregate" / "hybrid_selection.json"
        )
    else:
        assert selection is not None
        selected_teacher = str(selection["selected_teacher"])
        teacher_arm = (
            "PROTT5_TEACHER"
            if selected_teacher == "prott5"
            else "PROTT5_ESM2_TEACHER"
        )
        student_arm = (
            "PROTT5_DISTILLED_TRIPEPLITE"
            if selected_teacher == "prott5"
            else "PROTT5_ESM2_DISTILLED_TRIPEPLITE"
        )
        bootstrap = paired_student_bootstrap(
            data.labels[evaluation],
            score["TRIPEPLITE_HARD"],
            score[student_arm],
            data.groups[evaluation],
            config,
        )
        denominator = (
            metrics[teacher_arm]["roc_auc"] - metrics["B1_ORDER_LOGIT"]["roc_auc"]
        )
        retention = (
            None
            if denominator <= 0
            else (
                metrics[student_arm]["roc_auc"]
                - metrics["B1_ORDER_LOGIT"]["roc_auc"]
            )
            / denominator
        )
        aggregate["confirmation"] = {
            "selected_teacher_from_fold_0": selected_teacher,
            "primary_student_arm": student_arm,
            "primary_contrast": f"{student_arm} minus TRIPEPLITE_HARD",
            "paired_cluster_bootstrap": bootstrap,
            "teacher_gain_retention": retention,
            "retention_target": config["confirmation"][
                "teacher_gain_retention_target"
            ],
            "confirmation_fold_used_for_model_or_teacher_selection": False,
        }
    atomic_json(aggregate, run_root / "aggregate" / f"{stage}.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PLAN_ROOT / "config" / "DISTILLATION_CONFIG_20260813.json",
    )
    parser.add_argument(
        "--stage", choices=["preflight", "design", "confirmation"], required=True
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_root = Path(config["paths"]["private_run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        run_preflight(config, run_root)
    else:
        run_stage(config, run_root, args.stage)


if __name__ == "__main__":
    main()
