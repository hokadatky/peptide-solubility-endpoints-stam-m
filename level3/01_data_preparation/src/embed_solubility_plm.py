#!/usr/bin/env python3
"""Extract resumable, residue-only pLM embeddings for SoluProt sequences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch


MODELS = {
    "prott5": {
        "checkpoint": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "revision": "94a6abc029ae13029317b140b7424e012bf8dfbf",
        "dimension": 1024,
        "default_batch_tokens": 1800,
    },
    "esm2": {
        "checkpoint": "facebook/esm2_t33_650M_UR50D",
        "revision": "08e4846e537177426273712802403f7ba8261b6c",
        "dimension": 1280,
        "default_batch_tokens": 3600,
    },
}


@dataclass(frozen=True)
class Chunk:
    record_index: int
    chunk_index: int
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def load_records(train_path: Path, external_path: Path, limit: int) -> pd.DataFrame:
    train = pd.read_csv(train_path, dtype={"record_id": str, "source_id": str})
    external = pd.read_csv(external_path, dtype={"record_id": str, "source_id": str})
    train["analysis_split"] = "development"
    external["analysis_split"] = "external_test"
    columns = ["record_id", "source_id", "sequence", "length", "analysis_split"]
    records = pd.concat([train[columns], external[columns]], ignore_index=True)
    if records["record_id"].duplicated().any():
        raise ValueError("Record identifiers are not unique.")
    if limit:
        records = records.iloc[:limit].copy()
    if not np.array_equal(records["sequence"].str.len().to_numpy(), records["length"].to_numpy()):
        raise ValueError("Stored lengths and sequence strings disagree.")
    return records.reset_index(drop=True)


def make_chunks(records: pd.DataFrame, chunk_residues: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record_index, sequence in enumerate(records["sequence"]):
        for chunk_index, start in enumerate(range(0, len(sequence), chunk_residues)):
            chunks.append(
                Chunk(
                    record_index=record_index,
                    chunk_index=chunk_index,
                    sequence=sequence[start : start + chunk_residues],
                )
            )
    return sorted(chunks, key=lambda item: (-item.length, item.record_index, item.chunk_index))


def make_batches(chunks: list[Chunk], max_batch_tokens: int, max_batch_size: int) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_max = 0
    for chunk in chunks:
        proposed_max = max(current_max, chunk.length)
        proposed_size = len(current) + 1
        if current and (
            proposed_max * proposed_size > max_batch_tokens or proposed_size > max_batch_size
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(chunk)
        current_max = max(current_max, chunk.length)
    if current:
        batches.append(current)
    return batches


def load_model_and_tokenizer(model_key: str, device: str, dtype: torch.dtype):
    specification = MODELS[model_key]
    checkpoint = specification["checkpoint"]
    revision = specification["revision"]
    if model_key == "prott5":
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            do_lower_case=False,
            local_files_only=True,
        )
        model = T5EncoderModel.from_pretrained(
            checkpoint,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            checkpoint,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=True,
        )
    model = model.to(device).eval()
    return tokenizer, model


def tokenizer_inputs(model_key: str, sequences: list[str]) -> list[str]:
    if model_key == "prott5":
        return [" ".join(sequence) for sequence in sequences]
    return sequences


@torch.inference_mode()
def embed_batch(model_key: str, tokenizer, model, batch: list[Chunk], device: str) -> np.ndarray:
    sequences = [chunk.sequence for chunk in batch]
    encoded = tokenizer(
        tokenizer_inputs(model_key, sequences),
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_special_tokens_mask=True,
    )
    special = encoded.pop("special_tokens_mask")
    residue_mask = encoded["attention_mask"] * (1 - special)
    observed = residue_mask.sum(axis=1).cpu().numpy().astype(int)
    expected = np.asarray([chunk.length for chunk in batch], dtype=int)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"Tokenizer residue counts differ: observed={observed}, expected={expected}")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    residue_mask = residue_mask.to(device).unsqueeze(-1)
    hidden = model(**encoded).last_hidden_state
    sums = (hidden * residue_mask).sum(axis=1).float().cpu().numpy()
    synchronize(device)
    return sums


def memory_snapshot(device: str) -> dict[str, int | None]:
    process_rss = psutil.Process(os.getpid()).memory_info().rss
    accelerator_current: int | None = None
    accelerator_driver: int | None = None
    if device == "mps":
        accelerator_current = int(torch.mps.current_allocated_memory())
        accelerator_driver = int(torch.mps.driver_allocated_memory())
    elif device == "cuda":
        accelerator_current = int(torch.cuda.memory_allocated())
        accelerator_driver = int(torch.cuda.memory_reserved())
    return {
        "process_rss_bytes": int(process_rss),
        "accelerator_current_bytes": accelerator_current,
        "accelerator_driver_or_reserved_bytes": accelerator_driver,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/solubility/soluprot_train_partitions.csv"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/solubility/soluprot_external_test_clustered.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/solubility/embeddings"),
    )
    parser.add_argument("--chunk-residues", type=int, default=900)
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_key = args.model
    specification = MODELS[model_key]
    max_batch_tokens = args.max_batch_tokens or specification["default_batch_tokens"]
    if args.chunk_residues <= 0 or args.chunk_residues > 1000:
        raise ValueError("chunk-residues must be between 1 and 1,000.")

    train_path = args.train.resolve()
    external_path = args.external.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / f"{model_key}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(train_path, external_path, args.limit)
    chunks = make_chunks(records, args.chunk_residues)
    batches = make_batches(chunks, max_batch_tokens, args.max_batch_size)
    index_path = output_dir / "soluprot_embedding_index.csv"
    index = records[["record_id", "source_id", "length", "analysis_split"]].copy()
    index["chunk_count"] = [math.ceil(length / args.chunk_residues) for length in index["length"]]
    if index_path.exists():
        prior = pd.read_csv(index_path, dtype={"record_id": str, "source_id": str})
        if not prior.equals(index):
            raise ValueError(f"Existing embedding index is incompatible: {index_path}")
    else:
        index.to_csv(index_path, index=False)

    configuration = {
        "model_key": model_key,
        "checkpoint": specification["checkpoint"],
        "revision": specification["revision"],
        "chunk_residues": args.chunk_residues,
        "max_batch_tokens": max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "dtype": args.dtype,
        "n_records": len(records),
        "record_id_hash": stable_hash(records["record_id"].tolist()),
        "batch_schedule_hash": stable_hash(
            [[(chunk.record_index, chunk.chunk_index, chunk.length) for chunk in batch] for batch in batches]
        ),
    }
    config_hash = stable_hash(configuration)
    config_path = output_dir / f"soluprot_{model_key}_configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["configuration_hash"] != config_hash:
            raise ValueError(f"Existing shard configuration is incompatible: {config_path}")
    else:
        atomic_json({"configuration_hash": config_hash, **configuration}, config_path)

    final_path = output_dir / f"soluprot_{model_key}.npy"
    metadata_path = output_dir / f"soluprot_{model_key}_metadata.json"
    if final_path.exists() and metadata_path.exists():
        array = np.load(final_path, mmap_mode="r")
        if array.shape != (len(records), specification["dimension"]):
            raise ValueError(f"Existing final embedding has an incompatible shape: {array.shape}")
        print(f"Final embedding already exists: {final_path} {array.shape}")
        return

    expected_shards = [shard_dir / f"{index_number:06d}.npz" for index_number in range(len(batches))]
    missing_batch_indices = [i for i, path in enumerate(expected_shards) if not path.exists()]
    device = choose_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" and device != "cpu" else torch.float32

    load_seconds = 0.0
    warmup_seconds = 0.0
    extraction_seconds = 0.0
    peak = memory_snapshot(device)
    parameter_count: int | None = None
    completed_this_run = 0

    if missing_batch_indices:
        load_start = time.perf_counter()
        tokenizer, model = load_model_and_tokenizer(model_key, device, dtype)
        synchronize(device)
        load_seconds = time.perf_counter() - load_start
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

        warmup_batch = batches[missing_batch_indices[0]]
        warmup_start = time.perf_counter()
        _ = embed_batch(model_key, tokenizer, model, warmup_batch, device)
        synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start

        extraction_start = time.perf_counter()
        for position, batch_index in enumerate(missing_batch_indices, start=1):
            batch = batches[batch_index]
            sums = embed_batch(model_key, tokenizer, model, batch, device)
            atomic_npz(
                expected_shards[batch_index],
                configuration_hash=np.asarray([config_hash]),
                record_indices=np.asarray([chunk.record_index for chunk in batch], dtype=np.int32),
                chunk_indices=np.asarray([chunk.chunk_index for chunk in batch], dtype=np.int16),
                residue_counts=np.asarray([chunk.length for chunk in batch], dtype=np.int32),
                residue_sums=sums.astype(np.float32),
            )
            completed_this_run += 1
            snapshot = memory_snapshot(device)
            for key, value in snapshot.items():
                if value is not None:
                    prior = peak.get(key)
                    peak[key] = value if prior is None else max(int(prior), int(value))
            if position == 1 or position % 25 == 0 or position == len(missing_batch_indices):
                elapsed = time.perf_counter() - extraction_start
                completed_chunks = sum(len(batches[i]) for i in missing_batch_indices[:position])
                completed_residues = sum(
                    chunk.length for i in missing_batch_indices[:position] for chunk in batches[i]
                )
                print(
                    f"{model_key} batches {position}/{len(missing_batch_indices)}; "
                    f"chunks={completed_chunks}; residues/s={completed_residues / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )
        synchronize(device)
        extraction_seconds = time.perf_counter() - extraction_start
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    missing_after = [path for path in expected_shards if not path.exists()]
    if missing_after:
        raise RuntimeError(f"Embedding incomplete: {len(missing_after)} batch shards are missing.")

    dimension = int(specification["dimension"])
    residue_sums = np.zeros((len(records), dimension), dtype=np.float64)
    residue_counts = np.zeros(len(records), dtype=np.int64)
    for shard_path in expected_shards:
        with np.load(shard_path, allow_pickle=False) as shard:
            if str(shard["configuration_hash"][0]) != config_hash:
                raise ValueError(f"Configuration mismatch in shard: {shard_path}")
            indices = shard["record_indices"]
            np.add.at(residue_sums, indices, shard["residue_sums"].astype(np.float64))
            np.add.at(residue_counts, indices, shard["residue_counts"].astype(np.int64))
    if not np.array_equal(residue_counts, records["length"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Aggregated residue counts do not equal full sequence lengths.")
    embeddings = (residue_sums / residue_counts[:, None]).astype(np.float32)
    if embeddings.shape != (len(records), dimension) or not np.isfinite(embeddings).all():
        raise RuntimeError("Final embedding validation failed.")
    np.save(final_path, embeddings)

    total_residues = int(records["length"].sum())
    metadata = {
        "configuration_hash": config_hash,
        "configuration": configuration,
        "pooling": "residue-only arithmetic mean across every biological residue",
        "special_tokens_excluded": True,
        "chunk_context_limitation": (
            "Sequences longer than the chunk limit are encoded as non-overlapping blocks; "
            "context does not cross block boundaries."
        ),
        "device": device,
        "effective_torch_dtype": str(dtype).replace("torch.", ""),
        "parameter_count": parameter_count,
        "dimension": dimension,
        "n_records": int(len(records)),
        "n_chunks": int(len(chunks)),
        "n_batches": int(len(batches)),
        "total_residues": total_residues,
        "sequences_longer_than_chunk": int((records["length"] > args.chunk_residues).sum()),
        "timing_current_run_seconds": {
            "model_load": load_seconds,
            "warmup": warmup_seconds,
            "embedding": extraction_seconds,
        },
        "batches_completed_current_run": completed_this_run,
        "throughput_current_run": {
            "all_residues_per_second_if_uninterrupted": (
                total_residues / extraction_seconds if completed_this_run == len(batches) and extraction_seconds else None
            ),
            "all_sequences_per_second_if_uninterrupted": (
                len(records) / extraction_seconds if completed_this_run == len(batches) and extraction_seconds else None
            ),
        },
        "peak_memory_current_run": peak,
        "input_sha256": {
            "development": sha256(train_path),
            "external_test": sha256(external_path),
            "index": sha256(index_path),
        },
        "output_sha256": sha256(final_path),
        "output_bytes": final_path.stat().st_size,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "sentencepiece": package_version("sentencepiece"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "psutil": psutil.__version__,
        },
    }
    atomic_json(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
