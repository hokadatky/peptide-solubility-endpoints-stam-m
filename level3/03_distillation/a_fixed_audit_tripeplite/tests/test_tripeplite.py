from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PLAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN_ROOT / "src"))

from tripeplite import (  # noqa: E402
    AA_ALPHABET,
    EncodedTripeptideDataset,
    distillation_loss,
    encode_trimers,
    export_numpy_bundle,
    make_model,
    parameter_count,
    predict_numpy_bundle,
    predict_torch,
    train_fixed_student,
    validate_sequence,
)


def synthetic_sequences(n: int = 24) -> list[str]:
    rng = np.random.default_rng(20260813)
    return [
        "".join(rng.choice(list(AA_ALPHABET), size=12 + index % 17))
        for index in range(n)
    ]


def test_exact_trimer_encoding_is_bijective() -> None:
    trimers = ["".join(chars) for chars in itertools.product(AA_ALPHABET, repeat=3)]
    indices = np.asarray([encode_trimers(trimer)[0] for trimer in trimers])
    assert indices.min() == 0
    assert indices.max() == 7999
    assert len(np.unique(indices)) == 8000


def test_sequence_validation_and_short_sequence_contract() -> None:
    assert validate_sequence(" ac d\n") == "ACD"
    assert encode_trimers("A").tolist() == [8000]
    assert encode_trimers("AC").tolist() == [8000]
    with pytest.raises(ValueError, match="Non-canonical"):
        validate_sequence("ABZ")
    with pytest.raises(ValueError, match="empty"):
        validate_sequence("  ")


def test_fixed_parameter_count() -> None:
    assert parameter_count(make_model()) == 265633


def test_dataset_rejects_invalid_teacher_targets() -> None:
    sequences = synthetic_sequences(4)
    descriptors = np.zeros((4, 25), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly inside"):
        EncodedTripeptideDataset(
            sequences,
            descriptors,
            np.asarray([0, 1, 0, 1]),
            np.asarray([0.2, 0.0, 0.8, 1.0]),
        )


def test_hard_loss_is_exact_when_alpha_zero() -> None:
    import torch
    from torch.nn import functional

    logits = torch.tensor([-1.0, 0.5, 2.0])
    labels = torch.tensor([0.0, 1.0, 1.0])
    teacher = torch.tensor([0.1, 0.8, 0.9])
    total, hard, soft = distillation_loss(
        logits, labels, teacher, alpha=0.0, temperature=2.0
    )
    expected = functional.binary_cross_entropy_with_logits(logits, labels)
    assert torch.equal(total, expected)
    assert torch.equal(hard, expected)
    assert float(soft) == 0.0


def test_soft_loss_prefers_teacher_matching_logits() -> None:
    import torch

    teacher = torch.tensor([0.05, 0.25, 0.75, 0.95])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    matching = torch.logit(teacher)
    reversed_logits = -matching
    _, _, matching_soft = distillation_loss(
        matching, labels, teacher, alpha=1.0, temperature=2.0
    )
    _, _, reversed_soft = distillation_loss(
        reversed_logits, labels, teacher, alpha=1.0, temperature=2.0
    )
    assert float(matching_soft) < float(reversed_soft)


def test_training_is_reproducible_and_numpy_export_has_inference_parity(tmp_path) -> None:
    sequences = synthetic_sequences(24)
    rng = np.random.default_rng(11)
    descriptors = rng.normal(size=(24, 25)).astype(np.float32)
    labels = np.asarray([index % 2 for index in range(24)], dtype=np.float32)
    teacher = np.where(labels == 1, 0.8, 0.2).astype(np.float32)
    kwargs = dict(
        seed=20260813,
        epochs=2,
        batch_size=8,
        learning_rate=0.003,
        alpha=0.5,
        temperature=2.0,
        torch_threads=1,
    )
    first = train_fixed_student(sequences, descriptors, labels, teacher, **kwargs)
    second = train_fixed_student(sequences, descriptors, labels, teacher, **kwargs)
    first_score = predict_torch(first, sequences, descriptors)
    second_score = predict_torch(second, sequences, descriptors)
    assert np.array_equal(first_score, second_score)

    artifact = tmp_path / "tripeplite.npz"
    metadata = export_numpy_bundle(
        first,
        artifact,
        descriptor_names=[f"x{index}" for index in range(25)],
        endpoint="synthetic_test_only",
    )
    numpy_score = predict_numpy_bundle(artifact, sequences, descriptors)
    assert np.max(np.abs(first_score - numpy_score)) < 1e-6
    assert metadata["teacher_required_at_inference"] is False
    assert metadata["contains_teacher_outputs"] is False
    assert metadata["contains_sequences"] is False
    assert metadata["contains_labels"] is False
    assert metadata["contains_record_ids"] is False
    assert metadata["artifact_bytes"] < 5 * 1024 * 1024

    with np.load(artifact, allow_pickle=False) as bundle:
        keys = set(bundle.files)
        metadata_in_file = json.loads(bundle["metadata_json_utf8"].tobytes())
    forbidden_fragments = {"teacher_probability", "sequence", "label", "record_id"}
    assert not any(
        fragment in key.lower() for key in keys for fragment in forbidden_fragments
    )
    assert metadata_in_file["teacher_required_at_inference"] is False

