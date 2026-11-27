"""Bounded scheduler for distinct V4 checkpoint tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper/peptide_solubility_review"
WORKER = Path(__file__).with_name("v4_parallel_fold_worker.py")
sys.path.insert(0, str(PAPER))

from analysis_addon.src.run_soluprot_reanalysis import load_config  # noqa: E402


CONFIG = PAPER / "analysis_addon/config/analysis_config_v4.yaml"
PRINT_LOCK = threading.Lock()


def emit(payload: dict) -> None:
    with PRINT_LOCK:
        print(json.dumps(payload, sort_keys=True), flush=True)


def invoke(arguments: list[str], key: str) -> dict:
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "4"
    command = [sys.executable, "-B", str(WORKER), *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "key": key,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    emit(payload)
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed for {key}: {payload}")
    return payload


def stability_tasks(config: dict, arm: str) -> list[tuple[str, list[str]]]:
    tasks = []
    for seed in [int(value) for value in config["soluprot"]["fold_stability_seeds"]]:
        for outer_fold in range(int(config["soluprot"]["primary_outer_folds"])):
            key = f"stability:{seed}:{arm}:outer{outer_fold:02d}"
            tasks.append(
                (
                    key,
                    [
                        "--task",
                        "stability",
                        "--arm",
                        arm,
                        "--stability-seed",
                        str(seed),
                        "--outer-fold",
                        str(outer_fold),
                    ],
                )
            )
    return tasks


def negative_tasks(config: dict) -> list[tuple[str, list[str]]]:
    return [
        (
            f"negative-control:{repeat:03d}",
            ["--task", "negative-control", "--repeat", str(repeat)],
        )
        for repeat in range(int(config["soluprot"]["label_permutation_replicates"]))
    ]


def execute_pool(tasks: list[tuple[str, list[str]]], workers: int) -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(invoke, arguments, key): key for key, arguments in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["stability", "negative-control"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidate-workers", type=int, default=2)
    parser.add_argument("--reference-workers", type=int, default=1)
    parser.add_argument("--negative-workers", type=int, default=4)
    args = parser.parse_args()
    for name, value in (
        ("candidate-workers", args.candidate_workers),
        ("reference-workers", args.reference_workers),
        ("negative-workers", args.negative_workers),
    ):
        if value < 1 or value > 4:
            raise ValueError(f"{name} must be between 1 and 4.")

    config = load_config(CONFIG)
    if args.phase == "stability":
        candidate = stability_tasks(config, "candidate")
        reference = stability_tasks(config, "reference")
        keys = [key for key, _ in candidate + reference]
        if len(keys) != 70 or len(set(keys)) != 70:
            raise RuntimeError("Stability task set must contain 70 unique checkpoints.")
        emit(
            {
                "phase": "stability",
                "dry_run": bool(args.dry_run),
                "candidate_tasks": len(candidate),
                "reference_tasks": len(reference),
                "candidate_workers": args.candidate_workers,
                "reference_workers": args.reference_workers,
                "unique_tasks": len(set(keys)),
            }
        )
        if args.dry_run:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as controller:
            candidate_future = controller.submit(
                execute_pool, candidate, args.candidate_workers
            )
            reference_future = controller.submit(
                execute_pool, reference, args.reference_workers
            )
            candidate_results = candidate_future.result()
            reference_results = reference_future.result()
        emit(
            {
                "phase": "stability",
                "status": "completed",
                "candidate_completed": len(candidate_results),
                "reference_completed": len(reference_results),
            }
        )
        return

    tasks = negative_tasks(config)
    keys = [key for key, _ in tasks]
    if len(keys) != 50 or len(set(keys)) != 50:
        raise RuntimeError("Negative-control task set must contain 50 unique checkpoints.")
    emit(
        {
            "phase": "negative-control",
            "dry_run": bool(args.dry_run),
            "tasks": len(tasks),
            "workers": args.negative_workers,
            "unique_tasks": len(set(keys)),
        }
    )
    if args.dry_run:
        return
    results = execute_pool(tasks, args.negative_workers)
    emit(
        {
            "phase": "negative-control",
            "status": "completed",
            "completed": len(results),
        }
    )


if __name__ == "__main__":
    main()
