"""TriPepLite-v1: a teacher-free endpoint-specific peptide/protein student.

The training implementation uses PyTorch.  The exported inference bundle is a
plain ``.npz`` file and the NumPy inference path does not import PyTorch.  The
student receives only a canonical amino-acid sequence and 25 transparent
descriptors deterministically calculated from that sequence.  No PLM
embedding, label, record identifier, or teacher output is serialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ALPHABET)}
TRIMER_VOCABULARY = len(AA_ALPHABET) ** 3
SHORT_SEQUENCE_TOKEN = TRIMER_VOCABULARY
MODEL_VOCABULARY = TRIMER_VOCABULARY + 1


def validate_sequence(sequence: str) -> str:
    """Return an uppercase canonical sequence or fail closed."""
    if not isinstance(sequence, str):
        raise TypeError("Sequence must be a string.")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Sequence is empty.")
    invalid = sorted(set(normalized) - set(AA_ALPHABET))
    if invalid:
        raise ValueError(f"Non-canonical amino-acid symbols: {invalid}")
    return normalized


def encode_trimers(sequence: str) -> np.ndarray:
    """Encode exact overlapping 3-mers with no hashing collisions.

    The 8,000 canonical trimers map bijectively to ``0..7999``.  Sequences
    shorter than three residues map to one reserved token.  Such sequences are
    outside the observed SoluProt development range and must be flagged by the
    application-domain layer, but inference remains defined.
    """
    sequence = validate_sequence(sequence)
    if len(sequence) < 3:
        return np.asarray([SHORT_SEQUENCE_TOKEN], dtype=np.int64)
    encoded = np.empty(len(sequence) - 2, dtype=np.int64)
    for position in range(len(sequence) - 2):
        left = AA_TO_INDEX[sequence[position]]
        middle = AA_TO_INDEX[sequence[position + 1]]
        right = AA_TO_INDEX[sequence[position + 2]]
        encoded[position] = left * 400 + middle * 20 + right
    return encoded


@dataclass(frozen=True)
class DescriptorStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("Descriptor matrix has an incompatible shape.")
        transformed = (matrix - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Standardized descriptors contain non-finite values.")
        return transformed.astype(np.float32, copy=False)


def fit_descriptor_standardizer(matrix: np.ndarray) -> DescriptorStandardizer:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Descriptor fitting matrix is invalid.")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return DescriptorStandardizer(mean.astype(np.float32), scale.astype(np.float32))


class EncodedTripeptideDataset:
    """In-memory exact trimer encoding used by bounded student fits."""

    def __init__(
        self,
        sequences: Sequence[str],
        descriptors: np.ndarray,
        labels: np.ndarray | None = None,
        teacher_probabilities: np.ndarray | None = None,
        *,
        terminal_residues: int = 32,
    ) -> None:
        if terminal_residues < 3:
            raise ValueError("terminal_residues must be at least three.")
        self.sequences = [validate_sequence(str(sequence)) for sequence in sequences]
        self.descriptors = np.asarray(descriptors, dtype=np.float32)
        if self.descriptors.ndim != 2 or len(self.descriptors) != len(self.sequences):
            raise ValueError("Sequence/descriptor cardinality mismatch.")
        if not np.isfinite(self.descriptors).all():
            raise ValueError("Descriptors contain non-finite values.")
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        if self.labels is not None:
            if self.labels.shape != (len(self.sequences),):
                raise ValueError("Label cardinality mismatch.")
            if not set(np.unique(self.labels).tolist()).issubset({0.0, 1.0}):
                raise ValueError("Labels must be binary.")
        self.teacher_probabilities = (
            None
            if teacher_probabilities is None
            else np.asarray(teacher_probabilities, dtype=np.float32)
        )
        if self.teacher_probabilities is not None:
            if self.teacher_probabilities.shape != (len(self.sequences),):
                raise ValueError("Teacher-target cardinality mismatch.")
            if not np.isfinite(self.teacher_probabilities).all() or np.any(
                (self.teacher_probabilities <= 0.0) | (self.teacher_probabilities >= 1.0)
            ):
                raise ValueError("Teacher probabilities must be finite and strictly inside (0,1).")
        self.global_tokens: list[np.ndarray] = []
        self.n_terminal_tokens: list[np.ndarray] = []
        self.c_terminal_tokens: list[np.ndarray] = []
        for sequence in self.sequences:
            self.global_tokens.append(encode_trimers(sequence))
            self.n_terminal_tokens.append(encode_trimers(sequence[:terminal_residues]))
            self.c_terminal_tokens.append(encode_trimers(sequence[-terminal_residues:]))

    def __len__(self) -> int:
        return len(self.sequences)


def _packed_bag(arrays: Sequence[np.ndarray], indices: np.ndarray):
    import torch

    selected = [arrays[int(index)] for index in indices]
    lengths = np.asarray([len(array) for array in selected], dtype=np.int64)
    if np.any(lengths < 1):
        raise RuntimeError("Every embedding bag must contain at least one token.")
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    tokens = np.concatenate(selected).astype(np.int64, copy=False)
    return torch.from_numpy(tokens), torch.from_numpy(offsets)


def torch_batch(dataset: EncodedTripeptideDataset, indices: np.ndarray, device: str):
    import torch

    indices = np.asarray(indices, dtype=np.int64)
    global_tokens, global_offsets = _packed_bag(dataset.global_tokens, indices)
    n_tokens, n_offsets = _packed_bag(dataset.n_terminal_tokens, indices)
    c_tokens, c_offsets = _packed_bag(dataset.c_terminal_tokens, indices)
    descriptors = torch.from_numpy(dataset.descriptors[indices])
    labels = None if dataset.labels is None else torch.from_numpy(dataset.labels[indices])
    teacher = (
        None
        if dataset.teacher_probabilities is None
        else torch.from_numpy(dataset.teacher_probabilities[indices])
    )
    tensors = [
        global_tokens,
        global_offsets,
        n_tokens,
        n_offsets,
        c_tokens,
        c_offsets,
        descriptors,
    ]
    tensors = [tensor.to(device) for tensor in tensors]
    labels = None if labels is None else labels.to(device)
    teacher = None if teacher is None else teacher.to(device)
    return (*tensors, labels, teacher)


def make_model(
    *,
    descriptor_dimension: int = 25,
    embedding_dimension: int = 32,
    descriptor_hidden_dimension: int = 32,
    head_hidden_dimension: int = 64,
    dropout: float = 0.10,
):
    """Construct the fixed TriPepLite-v1 PyTorch module."""
    import torch
    from torch import nn

    class TriPepLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trimer_embedding = nn.EmbeddingBag(
                MODEL_VOCABULARY,
                embedding_dimension,
                mode="mean",
                include_last_offset=False,
            )
            self.global_norm = nn.LayerNorm(embedding_dimension)
            self.n_terminal_norm = nn.LayerNorm(embedding_dimension)
            self.c_terminal_norm = nn.LayerNorm(embedding_dimension)
            self.descriptor_projection = nn.Sequential(
                nn.Linear(descriptor_dimension, descriptor_hidden_dimension),
                nn.SiLU(),
            )
            combined_dimension = 3 * embedding_dimension + descriptor_hidden_dimension
            self.combined_norm = nn.LayerNorm(combined_dimension)
            self.head = nn.Sequential(
                nn.Linear(combined_dimension, head_hidden_dimension),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden_dimension, 1),
            )

        def forward(
            self,
            global_tokens,
            global_offsets,
            n_tokens,
            n_offsets,
            c_tokens,
            c_offsets,
            descriptors,
        ):
            global_vector = self.global_norm(
                self.trimer_embedding(global_tokens, global_offsets)
            )
            n_vector = self.n_terminal_norm(
                self.trimer_embedding(n_tokens, n_offsets)
            )
            c_vector = self.c_terminal_norm(
                self.trimer_embedding(c_tokens, c_offsets)
            )
            descriptor_vector = self.descriptor_projection(descriptors)
            combined = torch.cat(
                [global_vector, n_vector, c_vector, descriptor_vector], dim=1
            )
            return self.head(self.combined_norm(combined)).squeeze(1)

    return TriPepLite()


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def set_deterministic_seed(seed: int, *, torch_threads: int = 1) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(int(torch_threads))
    # CPU is the frozen primary device.  This call catches accidental use of a
    # nondeterministic kernel instead of silently changing the estimand.
    torch.use_deterministic_algorithms(True)


def distillation_loss(
    student_logits,
    labels,
    teacher_probabilities=None,
    *,
    alpha: float = 0.50,
    temperature: float = 2.0,
):
    """Binary hard/soft-target loss with standard temperature scaling."""
    import torch
    from torch.nn import functional as functional

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between zero and one.")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    hard = functional.binary_cross_entropy_with_logits(student_logits, labels)
    if teacher_probabilities is None or float(alpha) == 0.0:
        return hard, hard.detach(), torch.zeros_like(hard.detach())
    clipped = teacher_probabilities.clamp(1e-6, 1.0 - 1e-6)
    teacher_logits = torch.logit(clipped)
    softened_teacher = torch.sigmoid(teacher_logits / float(temperature))
    soft = functional.binary_cross_entropy_with_logits(
        student_logits / float(temperature), softened_teacher
    ) * float(temperature) ** 2
    total = (1.0 - float(alpha)) * hard + float(alpha) * soft
    return total, hard.detach(), soft.detach()


@dataclass
class FitResult:
    model: object
    standardizer: DescriptorStandardizer
    history: list[dict[str, float]]
    seed: int
    alpha: float
    temperature: float


def train_fixed_student(
    sequences: Sequence[str],
    descriptors: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray | None,
    *,
    seed: int = 20260813,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 0.003,
    weight_decay: float = 0.0001,
    gradient_clip_norm: float = 1.0,
    alpha: float = 0.50,
    temperature: float = 2.0,
    terminal_residues: int = 32,
    device: str = "cpu",
    torch_threads: int = 1,
) -> FitResult:
    """Fit one frozen student specification; no validation or checkpoint search."""
    import torch

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if teacher_probabilities is None and alpha != 0.0:
        raise ValueError("A non-zero alpha requires teacher probabilities.")
    set_deterministic_seed(seed, torch_threads=torch_threads)
    standardizer = fit_descriptor_standardizer(descriptors)
    transformed = standardizer.transform(descriptors)
    teacher = None
    if teacher_probabilities is not None:
        teacher = np.clip(
            np.asarray(teacher_probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6
        )
    dataset = EncodedTripeptideDataset(
        sequences,
        transformed,
        labels,
        teacher,
        terminal_residues=terminal_residues,
    )
    model = make_model(descriptor_dimension=transformed.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(weight_decay),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        generator = np.random.default_rng(int(seed) + epoch)
        order = generator.permutation(len(dataset))
        total_weighted = 0.0
        hard_weighted = 0.0
        soft_weighted = 0.0
        observed = 0
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch = torch_batch(dataset, indices, device)
            inputs, batch_labels, batch_teacher = batch[:7], batch[7], batch[8]
            if batch_labels is None:
                raise RuntimeError("Training batch has no labels.")
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss, hard, soft = distillation_loss(
                logits,
                batch_labels,
                batch_teacher,
                alpha=float(alpha),
                temperature=float(temperature),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Student loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            weight = len(indices)
            total_weighted += float(loss.detach().cpu()) * weight
            hard_weighted += float(hard.cpu()) * weight
            soft_weighted += float(soft.cpu()) * weight
            observed += weight
        history.append(
            {
                "epoch": float(epoch + 1),
                "total_loss": total_weighted / observed,
                "hard_loss": hard_weighted / observed,
                "soft_loss": soft_weighted / observed,
            }
        )
    return FitResult(
        model=model.cpu().eval(),
        standardizer=standardizer,
        history=history,
        seed=int(seed),
        alpha=float(alpha),
        temperature=float(temperature),
    )


def predict_torch(
    fit: FitResult,
    sequences: Sequence[str],
    descriptors: np.ndarray,
    *,
    terminal_residues: int = 32,
    batch_size: int = 256,
) -> np.ndarray:
    import torch

    transformed = fit.standardizer.transform(descriptors)
    dataset = EncodedTripeptideDataset(
        sequences, transformed, terminal_residues=terminal_residues
    )
    logits: list[np.ndarray] = []
    fit.model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), len(dataset)))
            batch = torch_batch(dataset, indices, "cpu")
            logits.append(fit.model(*batch[:7]).cpu().numpy())
    output = np.concatenate(logits).astype(np.float64)
    if output.shape != (len(dataset),) or not np.isfinite(output).all():
        raise RuntimeError("Student prediction failed output gates.")
    return 1.0 / (1.0 + np.exp(-np.clip(output, -50.0, 50.0)))


def _state_numpy(model) -> dict[str, np.ndarray]:
    state = model.state_dict()
    return {key: value.detach().cpu().numpy() for key, value in state.items()}


def export_numpy_bundle(
    fit: FitResult,
    path: Path,
    *,
    descriptor_names: Sequence[str],
    endpoint: str,
    terminal_residues: int = 32,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Export only inference weights and aggregate provenance to compressed NPZ."""
    path = Path(path)
    if len(descriptor_names) != len(fit.standardizer.mean):
        raise ValueError("Descriptor-name count does not match the fitted standardizer.")
    state = _state_numpy(fit.model)
    metadata: dict[str, object] = {
        "schema_version": "tripeplite-v1",
        "endpoint": endpoint,
        "alphabet": AA_ALPHABET,
        "trimer_vocabulary": TRIMER_VOCABULARY,
        "short_sequence_token": SHORT_SEQUENCE_TOKEN,
        "terminal_residues": int(terminal_residues),
        "descriptor_names": [str(name) for name in descriptor_names],
        "parameter_count": parameter_count(fit.model),
        "training_seed": int(fit.seed),
        "distillation_alpha": float(fit.alpha),
        "distillation_temperature": float(fit.temperature),
        "teacher_required_at_inference": False,
        "contains_teacher_outputs": False,
        "contains_training_rows": False,
        "contains_sequences": False,
        "contains_labels": False,
        "contains_record_ids": False,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    arrays: dict[str, np.ndarray] = {
        "metadata_json_utf8": np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
        ),
        "descriptor_mean": fit.standardizer.mean,
        "descriptor_scale": fit.standardizer.scale,
    }
    arrays.update(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**metadata, "artifact_sha256": digest, "artifact_bytes": path.stat().st_size}


def load_numpy_bundle(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        arrays = {key: np.asarray(bundle[key]) for key in bundle.files}
    metadata = json.loads(arrays.pop("metadata_json_utf8").tobytes().decode("utf-8"))
    if metadata.get("schema_version") != "tripeplite-v1":
        raise ValueError("Unsupported TriPepLite bundle schema.")
    return metadata, arrays


def _layer_norm(vector: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = vector.mean(axis=1, keepdims=True)
    variance = np.square(vector - mean).mean(axis=1, keepdims=True)
    return (vector - mean) / np.sqrt(variance + 1e-5) * weight + bias


def _silu(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -50.0, 50.0)
    return value / (1.0 + np.exp(-clipped))


def _mean_embedding(table: np.ndarray, token_lists: Sequence[np.ndarray]) -> np.ndarray:
    return np.vstack([table[tokens].mean(axis=0) for tokens in token_lists])


def predict_numpy_bundle(
    path: Path,
    sequences: Sequence[str],
    descriptors: np.ndarray,
) -> np.ndarray:
    """Teacher-free, PyTorch-free probability inference."""
    metadata, arrays = load_numpy_bundle(path)
    return predict_numpy_state(metadata, arrays, sequences, descriptors)


def predict_numpy_state(
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
    sequences: Sequence[str],
    descriptors: np.ndarray,
) -> np.ndarray:
    """Teacher-free inference from an already-loaded bundle."""
    terminal_residues = int(metadata["terminal_residues"])
    sequences = [validate_sequence(str(sequence)) for sequence in sequences]
    descriptor_matrix = np.asarray(descriptors, dtype=np.float32)
    descriptor_matrix = (
        descriptor_matrix - arrays["descriptor_mean"]
    ) / arrays["descriptor_scale"]
    table = arrays["trimer_embedding.weight"]
    global_tokens = [encode_trimers(sequence) for sequence in sequences]
    n_tokens = [encode_trimers(sequence[:terminal_residues]) for sequence in sequences]
    c_tokens = [encode_trimers(sequence[-terminal_residues:]) for sequence in sequences]
    global_vector = _layer_norm(
        _mean_embedding(table, global_tokens),
        arrays["global_norm.weight"],
        arrays["global_norm.bias"],
    )
    n_vector = _layer_norm(
        _mean_embedding(table, n_tokens),
        arrays["n_terminal_norm.weight"],
        arrays["n_terminal_norm.bias"],
    )
    c_vector = _layer_norm(
        _mean_embedding(table, c_tokens),
        arrays["c_terminal_norm.weight"],
        arrays["c_terminal_norm.bias"],
    )
    descriptor_vector = _silu(
        descriptor_matrix @ arrays["descriptor_projection.0.weight"].T
        + arrays["descriptor_projection.0.bias"]
    )
    combined = np.concatenate(
        [global_vector, n_vector, c_vector, descriptor_vector], axis=1
    )
    combined = _layer_norm(
        combined, arrays["combined_norm.weight"], arrays["combined_norm.bias"]
    )
    hidden = _silu(
        combined @ arrays["head.0.weight"].T + arrays["head.0.bias"]
    )
    logits = hidden @ arrays["head.3.weight"].T + arrays["head.3.bias"]
    logits = logits[:, 0].astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    if probabilities.shape != (len(sequences),) or not np.isfinite(probabilities).all():
        raise RuntimeError("NumPy inference failed output gates.")
    return probabilities

