#!/usr/bin/env python3
"""Bounded synthetic and tiny-real-data smoke tests; no scientific estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True


PLAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAN_ROOT.parents[1]
V4_SRC = (
    PROJECT_ROOT / "paper" / "peptide_solubility_review" / "analysis_addon" / "src"
)
sys.path.insert(0, str(PLAN_ROOT / "src"))
sys.path.insert(0, str(V4_SRC))

from nested_models import fit_head, predict_head  # noqa: E402
from sequence_features import build_b0_b1, primary_feature_names  # noqa: E402
from tripeplite import (  # noqa: E402
    AA_ALPHABET,
    export_numpy_bundle,
    make_model,
    parameter_count,
    predict_numpy_bundle,
    predict_torch,
    train_fixed_student,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_data(n: int = 96):
    rng = np.random.default_rng(20260813)
    sequences: list[str] = []
    descriptors = rng.normal(size=(n, 25)).astype(np.float32)
    latent = np.zeros(n, dtype=np.float64)
    for index in range(n):
        length = int(rng.integers(18, 91))
        sequence = "".join(rng.choice(list(AA_ALPHABET), size=length))
        sequences.append(sequence)
        latent[index] = (
            1.1 * sequence.count("K")
            + 0.9 * sequence.count("R")
            - 0.8 * sequence.count("W")
            - 0.6 * sequence.count("F")
        ) / np.sqrt(length) + 0.35 * descriptors[index, 0]
    labels = (latent > np.median(latent)).astype(np.float32)
    teacher = 1.0 / (1.0 + np.exp(-1.8 * latent))
    teacher = np.clip(teacher, 0.02, 0.98).astype(np.float32)
    return sequences, descriptors, labels, teacher


def tiny_real_data(n_per_class: int = 32):
    development_path = PROJECT_ROOT / "data/solubility/soluprot_train_partitions.csv"
    feature_path = (
        PROJECT_ROOT / "data/solubility/solubility_transparent_features_development.csv"
    )
    manifest_path = PROJECT_ROOT / "data/solubility/solubility_transparent_features.json"
    development = pd.read_csv(development_path, dtype={"record_id": str})
    features = pd.read_csv(feature_path, dtype={"record_id": str})
    if not np.array_equal(development.record_id.to_numpy(), features.record_id.to_numpy()):
        raise RuntimeError("Development/features row alignment failed.")
    eligible = development[development.development_fold.isin([2, 3, 4, 5, 6])].copy()
    eligible = eligible.drop_duplicates("cluster_id", keep="first")
    selected_indices: list[int] = []
    for label in (0, 1):
        candidates = eligible[eligible.label == label].sort_values(
            ["development_fold", "length", "record_id"], kind="mergesort"
        )
        positions = np.linspace(0, len(candidates) - 1, n_per_class, dtype=int)
        selected_indices.extend(candidates.iloc[positions].index.tolist())
    names = primary_feature_names(manifest_path)
    blocks = build_b0_b1(
        features.loc[selected_indices],
        development.loc[selected_indices, "sequence"].tolist(),
        manifest_path,
    )
    return (
        development.loc[selected_indices, "sequence"].tolist(),
        blocks["B0"][0].astype(np.float32),
        blocks["B1"][0].astype(np.float32),
        development.loc[selected_indices, "label"].to_numpy(dtype=np.float32),
        names,
        development.loc[selected_indices, "length"].to_numpy(dtype=int),
        np.asarray(selected_indices, dtype=int),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    sequences, descriptors, labels, teacher = synthetic_data()
    fit = train_fixed_student(
        sequences,
        descriptors,
        labels,
        teacher,
        seed=20260813,
        epochs=5,
        batch_size=32,
        alpha=0.5,
        temperature=2.0,
        torch_threads=1,
    )
    synthetic_score = predict_torch(fit, sequences, descriptors)
    with tempfile.TemporaryDirectory(prefix="tripeplite_dryrun_") as temporary:
        artifact = Path(temporary) / "synthetic_tripeplite.npz"
        metadata = export_numpy_bundle(
            fit,
            artifact,
            descriptor_names=[f"synthetic_x{index}" for index in range(25)],
            endpoint="synthetic_dry_run_only",
        )
        numpy_score = predict_numpy_bundle(artifact, sequences, descriptors)
        parity_max_abs = float(np.max(np.abs(synthetic_score - numpy_score)))
        artifact_hash = sha256(artifact)
        artifact_bytes = artifact.stat().st_size

    (
        real_sequences,
        real_descriptors,
        real_b1,
        real_labels,
        descriptor_names,
        lengths,
        source_indices,
    ) = tiny_real_data()
    real_fit = train_fixed_student(
        real_sequences,
        real_descriptors,
        real_labels,
        None,
        seed=20260813,
        epochs=2,
        batch_size=32,
        alpha=0.0,
        temperature=2.0,
        torch_threads=1,
    )
    real_score = predict_torch(real_fit, real_sequences, real_descriptors)

    # One fixed-head plumbing check on 64 real rows.  The split is constructed
    # by class blocks (24+24 train, 8+8 score) and no metric is calculated.
    teacher_train = np.r_[0:24, 32:56]
    teacher_score_indices = np.r_[24:32, 56:64]
    prott5 = np.load(
        PROJECT_ROOT / "data/solubility/embeddings/soluprot_prott5.npy", mmap_mode="r"
    )
    small_embedding = np.asarray(prott5[source_indices], dtype=np.float32)
    teacher_start = time.perf_counter()
    teacher_head = fit_head(
        real_b1[teacher_train],
        small_embedding[teacher_train],
        real_labels[teacher_train].astype(int),
        "pls",
        c_value=1000.0,
        pls_k=8,
    )
    small_teacher_score = predict_head(
        teacher_head,
        real_b1[teacher_score_indices],
        small_embedding[teacher_score_indices],
    )
    teacher_elapsed = time.perf_counter() - teacher_start

    payload = {
        "status": "PASS",
        "scope": "engineering dry run only; no scientific performance estimate",
        "student": {
            "name": "TriPepLite-v1",
            "parameter_count": parameter_count(make_model()),
            "teacher_required_at_inference": False,
            "numpy_artifact_bytes": int(artifact_bytes),
            "numpy_artifact_sha256_ephemeral": artifact_hash,
            "torch_numpy_probability_max_abs_difference": parity_max_abs,
        },
        "synthetic": {
            "rows": len(sequences),
            "epochs": 5,
            "initial_total_loss": fit.history[0]["total_loss"],
            "final_total_loss": fit.history[-1]["total_loss"],
            "loss_finite": bool(
                np.isfinite([row["total_loss"] for row in fit.history]).all()
            ),
            "output_finite_and_in_range": bool(
                np.isfinite(synthetic_score).all()
                and np.all((synthetic_score >= 0.0) & (synthetic_score <= 1.0))
            ),
        },
        "tiny_real": {
            "rows": len(real_sequences),
            "unique_groups_required_by_selection": True,
            "source_folds": [2, 3, 4, 5, 6],
            "epochs": 2,
            "observed_length_min": int(lengths.min()),
            "observed_length_max": int(lengths.max()),
            "descriptor_dimension": len(descriptor_names),
            "loss_finite": bool(
                np.isfinite([row["total_loss"] for row in real_fit.history]).all()
            ),
            "output_shape_correct": bool(real_score.shape == (len(real_sequences),)),
            "performance_metrics_computed": False,
            "row_level_material_written": False,
        },
        "fixed_teacher_plumbing": {
            "teacher": "B1 plus ProtT5 PLS-8 fixed L2 logistic",
            "fit_rows": int(len(teacher_train)),
            "score_rows": int(len(teacher_score_indices)),
            "output_finite_and_in_range": bool(
                np.isfinite(small_teacher_score).all()
                and np.all((small_teacher_score > 0.0) & (small_teacher_score < 1.0))
            ),
            "performance_metrics_computed": False,
            "elapsed_seconds": teacher_elapsed,
        },
        "privacy": {
            "contains_record_ids": False,
            "contains_sequences": False,
            "contains_labels": False,
            "contains_row_level_predictions": False,
            "contains_teacher_targets": False,
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    if not (
        payload["student"]["torch_numpy_probability_max_abs_difference"] < 1e-6
        and payload["student"]["numpy_artifact_bytes"] < 5 * 1024 * 1024
        and payload["synthetic"]["loss_finite"]
        and payload["synthetic"]["output_finite_and_in_range"]
        and payload["tiny_real"]["loss_finite"]
        and payload["tiny_real"]["output_shape_correct"]
        and payload["fixed_teacher_plumbing"]["output_finite_and_in_range"]
    ):
        payload["status"] = "FAIL"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(encoded, encoding="utf-8")
    temporary_output.replace(args.output)
    print(encoded, end="")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
