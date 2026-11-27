"""Bounded scheduler for independent frozen V4 shift-audit checkpoints."""

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
WORKER = Path(__file__).with_name("v4_shift_parallel_worker.py")
CONFIG = PAPER / "analysis_addon/config/analysis_config_v4.yaml"
sys.path.insert(0, str(PAPER))

from analysis_addon.src.run_soluprot_reanalysis import load_config  # noqa: E402


PRINT_LOCK = threading.Lock()


def emit(payload: dict) -> None:
    with PRINT_LOCK:
        print(json.dumps(payload, sort_keys=True), flush=True)


def invoke(arguments: list[str], key: str) -> dict:
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [sys.executable, "-B", str(WORKER), *arguments],
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


def task_sets(config: dict) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    nonpls: list[tuple[str, list[str]]] = []
    pls: list[tuple[str, list[str]]] = []
    for threshold in [float(value) for value in config["soluprot"]["threshold_ladder"]]:
        for outer_fold in range(int(config["soluprot"]["primary_outer_folds"])):
            nonpls.append(
                (
                    f"ladder:{threshold:.2f}:reference:outer{outer_fold:02d}",
                    ["--task", "ladder", "--threshold", str(threshold), "--arm", "reference", "--outer-fold", str(outer_fold)],
                )
            )
            pls.append(
                (
                    f"ladder:{threshold:.2f}:candidate:outer{outer_fold:02d}",
                    ["--task", "ladder", "--threshold", str(threshold), "--arm", "candidate", "--outer-fold", str(outer_fold)],
                )
            )
    for arm in ("B0_transparent", "B1_order_aware"):
        for outer_fold in range(int(config["shift"]["source_classifier_folds"])):
            nonpls.append(
                (
                    f"source:{arm}:outer{outer_fold:02d}",
                    ["--task", "source", "--arm", arm, "--outer-fold", str(outer_fold)],
                )
            )
    for outer_fold in range(int(config["shift"]["source_classifier_folds"])):
        pls.append(
            (
                f"source:ProtT5_PLS:outer{outer_fold:02d}",
                ["--task", "source", "--arm", "ProtT5_PLS", "--outer-fold", str(outer_fold)],
            )
        )
    return nonpls, pls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cluster-prep", "nonpls", "pls"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4.")
    config = load_config(CONFIG)
    if args.phase == "cluster-prep":
        emit({"phase": args.phase, "dry_run": bool(args.dry_run), "tasks": 1})
        if not args.dry_run:
            invoke(["--task", "cluster-prep"], "cluster-prep")
        return
    nonpls, pls = task_sets(config)
    all_keys = [key for key, _ in nonpls + pls]
    if len(nonpls) != 38 or len(pls) != 33 or len(set(all_keys)) != 71:
        raise RuntimeError("Shift task inventory must be 38 non-PLS + 33 PLS = 71 unique tasks.")
    tasks = nonpls if args.phase == "nonpls" else pls
    emit(
        {
            "phase": args.phase,
            "dry_run": bool(args.dry_run),
            "tasks": len(tasks),
            "all_unique_tasks": len(set(all_keys)),
            "workers": args.workers,
        }
    )
    if args.dry_run:
        return
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(invoke, arguments, key): key for key, arguments in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    emit({"phase": args.phase, "status": "completed", "completed": len(results)})


if __name__ == "__main__":
    main()
