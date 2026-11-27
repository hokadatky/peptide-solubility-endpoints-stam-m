#!/usr/bin/env python3
"""Recompute one existing ESM-2 shard on CPU and audit numerical parity.

This bounded audit is required when the historical per-shard device provenance
is absent and extraction may be resumed on CPU/float32.  It writes aggregate
differences only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


sys.dont_write_bytecode = True
PLAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAN_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embed_solubility_plm import (  # noqa: E402
    embed_batch,
    load_model_and_tokenizer,
    load_records,
    make_batches,
    make_chunks,
)


def atomic_json(payload: object, path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == encoded:
            return
        raise FileExistsError(f"Refusing to overwrite a different parity audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=776)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = PROJECT_ROOT / "data/solubility/soluprot_train_partitions.csv"
    external = PROJECT_ROOT / "data/solubility/soluprot_external_test_clustered.csv"
    shard_path = (
        PROJECT_ROOT
        / "data/solubility/embeddings/esm2_shards"
        / f"{int(args.shard_index):06d}.npz"
    )
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    records = load_records(train, external, 0)
    batches = make_batches(make_chunks(records, 900), 3600, 32)
    if args.shard_index < 0 or args.shard_index >= len(batches):
        raise ValueError("Shard index is outside the frozen batch schedule.")
    batch = batches[int(args.shard_index)]
    with np.load(shard_path, allow_pickle=False) as shard:
        existing_indices = shard["record_indices"].astype(np.int64)
        existing_chunks = shard["chunk_indices"].astype(np.int64)
        existing_counts = shard["residue_counts"].astype(np.int64)
        existing_sums = shard["residue_sums"].astype(np.float64)
    expected_indices = np.asarray([item.record_index for item in batch])
    expected_chunks = np.asarray([item.chunk_index for item in batch])
    expected_counts = np.asarray([item.length for item in batch])
    if not (
        np.array_equal(existing_indices, expected_indices)
        and np.array_equal(existing_chunks, expected_chunks)
        and np.array_equal(existing_counts, expected_counts)
    ):
        raise ValueError("Existing shard does not match the frozen batch schedule.")

    started = time.perf_counter()
    tokenizer, model = load_model_and_tokenizer("esm2", "cpu", torch.float32)
    recomputed_sums = embed_batch("esm2", tokenizer, model, batch, "cpu").astype(
        np.float64
    )
    elapsed = time.perf_counter() - started
    denominator = existing_counts[:, None].astype(np.float64)
    existing_mean = existing_sums / denominator
    recomputed_mean = recomputed_sums / denominator
    absolute = np.abs(existing_mean - recomputed_mean)
    cosine = np.sum(existing_mean * recomputed_mean, axis=1) / (
        np.linalg.norm(existing_mean, axis=1)
        * np.linalg.norm(recomputed_mean, axis=1)
    )
    thresholds = {
        "minimum_cosine": 0.99999,
        "maximum_absolute_component_difference": 0.02,
        "maximum_p99_absolute_component_difference": 0.005,
    }
    observed = {
        "minimum_cosine": float(np.min(cosine)),
        "maximum_absolute_component_difference": float(np.max(absolute)),
        "p99_absolute_component_difference": float(np.percentile(absolute, 99)),
        "median_absolute_component_difference": float(np.median(absolute)),
    }
    checks = {
        "cosine_pass": observed["minimum_cosine"] >= thresholds["minimum_cosine"],
        "maximum_absolute_pass": observed["maximum_absolute_component_difference"]
        <= thresholds["maximum_absolute_component_difference"],
        "p99_absolute_pass": observed["p99_absolute_component_difference"]
        <= thresholds["maximum_p99_absolute_component_difference"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "one frozen existing shard recomputed on CPU float32",
        "shard_index": int(args.shard_index),
        "batch_rows": len(batch),
        "residue_counts": existing_counts.astype(int).tolist(),
        "checkpoint": "facebook/esm2_t33_650M_UR50D",
        "revision": "08e4846e537177426273712802403f7ba8261b6c",
        "recompute_device": "cpu",
        "recompute_dtype": "float32",
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "elapsed_seconds": elapsed,
        "privacy": {
            "contains_record_ids": False,
            "contains_sequences": False,
            "contains_labels": False,
            "contains_embeddings": False,
        },
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
