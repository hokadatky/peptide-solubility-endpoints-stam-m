#!/usr/bin/env python3
"""Benchmark final TriPepLite end-to-end CPU inference with no row-level data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


sys.dont_write_bytecode = True


PLAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAN_ROOT.parents[1]
sys.path.insert(0, str(PLAN_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from scoring import extract_sequence_features  # noqa: E402
from tripeplite import load_numpy_bundle, predict_numpy_state  # noqa: E402


def atomic_json(payload: object, path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == encoded:
            return
        raise FileExistsError(f"Refusing to overwrite a different benchmark: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 20:
        raise ValueError("Use at least one warmup and 20 measured repeats.")

    load_start = time.perf_counter_ns()
    metadata, arrays = load_numpy_bundle(args.artifact)
    load_ms = (time.perf_counter_ns() - load_start) / 1e6
    descriptor_names = [str(name) for name in metadata["descriptor_names"]]
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * 25)[:500]

    def score_once() -> float:
        features = extract_sequence_features(sequence)
        descriptor = np.asarray(
            [[features[name] for name in descriptor_names]], dtype=np.float32
        )
        return float(predict_numpy_state(metadata, arrays, [sequence], descriptor)[0])

    for _ in range(args.warmup):
        score_once()
    times_ms: list[float] = []
    scores: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter_ns()
        scores.append(score_once())
        times_ms.append((time.perf_counter_ns() - started) / 1e6)
    times = np.asarray(times_ms, dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(scores).all():
        raise RuntimeError("Benchmark produced a non-finite value.")
    if np.max(scores) - np.min(scores) > 1e-12:
        raise RuntimeError("Repeated deterministic inference changed its score.")

    payload = {
        "status": "PASS",
        "scope": "end-to-end descriptor calculation plus loaded NumPy student inference",
        "artifact": {
            "sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
            "bytes": args.artifact.stat().st_size,
            "schema_version": metadata["schema_version"],
            "parameter_count": metadata["parameter_count"],
            "teacher_required_at_inference": metadata["teacher_required_at_inference"],
        },
        "input": {
            "length": len(sequence),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "descriptor_dimension": len(descriptor_names),
        },
        "timing_ms": {
            "one_time_artifact_load": load_ms,
            "median_end_to_end": float(np.median(times)),
            "p95_end_to_end": float(np.percentile(times, 95)),
            "minimum_end_to_end": float(np.min(times)),
            "maximum_end_to_end": float(np.max(times)),
        },
        "warmup_repeats": int(args.warmup),
        "measured_repeats": int(args.repeats),
        "cpu_platform": platform.platform(),
        "python": sys.version.split()[0],
        "latency_target_median_ms": 100.0,
        "latency_target_pass": bool(float(np.median(times)) <= 100.0),
        "privacy": {
            "contains_record_ids": False,
            "contains_training_sequences": False,
            "contains_labels": False,
            "contains_row_level_predictions": False,
        },
    }
    if not payload["latency_target_pass"]:
        payload["status"] = "FAIL_LATENCY_GATE"
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
