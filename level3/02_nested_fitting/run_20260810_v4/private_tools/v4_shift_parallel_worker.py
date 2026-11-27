"""Compute one frozen V4 shift-audit checkpoint without changing its model path.

This operational worker imports the frozen implementation for data loading,
MMseqs grouping, split allocation, nested selection, fitting, prediction, and
atomic checkpoint validation.  The canonical shift-audit command must still be
run after all tasks; it reloads every artifact and is the only process allowed
to publish the aggregate result and stage manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper/peptide_solubility_review"
PRIVATE_TOOLS = Path(__file__).resolve().parent
RUN_DIR = ROOT / "paper/general_peptide_solubility_scoping/analysis_addon_run_20260810_v4"
CONFIG = PAPER / "analysis_addon/config/analysis_config_v4.yaml"
sys.path.insert(0, str(PAPER))
sys.path.insert(0, str(PRIVATE_TOOLS))

from analysis_addon.src.run_soluprot_reanalysis import load_config  # noqa: E402
from analysis_addon.src.run_soluprot_shift_audit import _load_core, _run_mmseqs  # noqa: E402
from analysis_addon.src.split_contracts import balanced_group_folds, fold_hash  # noqa: E402
from v4_parallel_fold_worker import compute_nested_fold  # noqa: E402


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_external_stage() -> None:
    manifest = RUN_DIR / "manifests/external-retrospective.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "External retrospective manifest is required before shift model tasks."
        )


def prepare_groups(config: dict, core: dict[str, object]) -> None:
    train = core["train"]
    outputs: dict[str, object] = {}
    for threshold in [float(value) for value in config["soluprot"]["threshold_ladder"]]:
        groups = _run_mmseqs(
            RUN_DIR,
            "development",
            train["sequence"].tolist(),
            threshold,
            config,
        )
        outputs[f"development_{threshold:.2f}"] = {
            "n_rows": int(len(groups)),
            "n_groups": int(len(np.unique(groups))),
            "group_array_sha256": _array_sha256(np.asarray(groups).astype("U")),
        }
    pooled_groups = _run_mmseqs(
        RUN_DIR,
        "pooled",
        core["all_source"]["sequence"].tolist(),
        0.30,
        config,
    )
    outputs["pooled_0.30"] = {
        "n_rows": int(len(pooled_groups)),
        "n_groups": int(len(np.unique(pooled_groups))),
        "group_array_sha256": _array_sha256(np.asarray(pooled_groups).astype("U")),
    }
    print(json.dumps({"status": "group_checkpoints_ready", "groups": outputs}, sort_keys=True))


def _require_group_checkpoint(name: str, threshold: float) -> None:
    path = RUN_DIR / "checkpoints/mmseqs" / f"{name}_{threshold:.2f}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Run cluster-prep first: {path}")


def compute_ladder(
    config: dict,
    core: dict[str, object],
    threshold: float,
    arm: str,
    outer_fold: int,
) -> None:
    frozen_thresholds = [float(value) for value in config["soluprot"]["threshold_ladder"]]
    if threshold not in frozen_thresholds:
        raise ValueError(f"Threshold is not frozen: {threshold}")
    if arm not in {"reference", "candidate"}:
        raise ValueError("Ladder arm must be reference or candidate.")
    _require_group_checkpoint("development", threshold)
    train = core["train"]
    labels = train["label"].to_numpy().astype(int)
    groups = _run_mmseqs(
        RUN_DIR, "development", train["sequence"].tolist(), threshold, config
    )
    folds = balanced_group_folds(
        labels,
        groups,
        int(config["soluprot"]["primary_outer_folds"]),
        int(config["global_seed"]),
    )
    if arm == "reference":
        embedding = None
        mode = "none"
        checkpoint_name = "B1_ORDER_LOGIT"
    else:
        embedding = np.asarray(core["embeddings"][: len(train)], dtype=np.float64)
        mode = "pls"
        checkpoint_name = "B1_ORDER_PLUS_PROTT5_PLS_LOGIT"
    threshold_name = f"threshold_{threshold:.2f}"
    compute_nested_fold(
        base=np.asarray(core["b1"][: len(train)], dtype=np.float64),
        embedding=embedding,
        y=labels,
        groups=groups,
        outer_folds=folds,
        mode=mode,
        c_grid=[float(value) for value in config["soluprot"]["logistic_c_grid"]],
        k_grid=[int(value) for value in config["soluprot"]["pls_k_grid"]],
        inner_folds=int(config["soluprot"]["inner_folds"]),
        seed=int(config["global_seed"]),
        checkpoint=(
            RUN_DIR
            / "checkpoints/shift/threshold_ladder"
            / threshold_name
            / arm
            / f"{checkpoint_name}_outer{outer_fold:02d}.npz"
        ),
        checkpoint_name=checkpoint_name,
        contract_tag=f"{config['run_id']}:shift:{threshold_name}:{arm}",
        outer_fold=outer_fold,
    )
    print(
        json.dumps(
            {
                "status": "ladder_fold_ready",
                "threshold": threshold,
                "arm": arm,
                "outer_fold": outer_fold,
                "fold_hash": fold_hash(groups, folds),
            },
            sort_keys=True,
        )
    )


def compute_source(
    config: dict,
    core: dict[str, object],
    arm: str,
    outer_fold: int,
) -> None:
    specs = {
        "B0_transparent": (np.asarray(core["b0"], dtype=np.float64), None, "none"),
        "B1_order_aware": (np.asarray(core["b1"], dtype=np.float64), None, "none"),
        "ProtT5_PLS": (
            np.zeros((len(core["all_source"]), 0), dtype=np.float64),
            np.asarray(core["embeddings"], dtype=np.float64),
            "pls",
        ),
    }
    if arm not in specs:
        raise ValueError(f"Unknown source-classifier arm: {arm}")
    _require_group_checkpoint("pooled", 0.30)
    labels = np.asarray(
        [0] * len(core["train"]) + [1] * len(core["external"]), dtype=int
    )
    groups = _run_mmseqs(
        RUN_DIR,
        "pooled",
        core["all_source"]["sequence"].tolist(),
        0.30,
        config,
    )
    folds = balanced_group_folds(
        labels,
        groups,
        int(config["shift"]["source_classifier_folds"]),
        int(config["global_seed"]),
    )
    base, embedding, mode = specs[arm]
    compute_nested_fold(
        base=base,
        embedding=embedding,
        y=labels,
        groups=groups,
        outer_folds=folds,
        mode=mode,
        c_grid=[float(value) for value in config["soluprot"]["logistic_c_grid"]],
        k_grid=[int(value) for value in config["soluprot"]["pls_k_grid"]],
        inner_folds=5,
        seed=int(config["global_seed"]),
        checkpoint=(
            RUN_DIR
            / "checkpoints/shift/source_classifier"
            / arm
            / f"{arm}_outer{outer_fold:02d}.npz"
        ),
        checkpoint_name=arm,
        contract_tag=f"{config['run_id']}:shift:source-classifier:{arm}",
        outer_fold=outer_fold,
    )
    print(
        json.dumps(
            {
                "status": "source_fold_ready",
                "arm": arm,
                "outer_fold": outer_fold,
                "fold_hash": fold_hash(groups, folds),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cluster-prep", "ladder", "source"], required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--arm")
    parser.add_argument("--outer-fold", type=int)
    args = parser.parse_args()
    config = load_config(CONFIG)
    core = _load_core(config)
    if args.task == "cluster-prep":
        prepare_groups(config, core)
        return
    _require_external_stage()
    if args.outer_fold is None:
        raise ValueError("--outer-fold is required for a model task.")
    if args.task == "ladder":
        if args.threshold is None or not 0 <= args.outer_fold < int(config["soluprot"]["primary_outer_folds"]):
            raise ValueError("A frozen threshold and valid seven-fold index are required.")
        compute_ladder(config, core, float(args.threshold), str(args.arm), int(args.outer_fold))
        return
    if not 0 <= args.outer_fold < int(config["shift"]["source_classifier_folds"]):
        raise ValueError("A valid five-fold source-classifier index is required.")
    compute_source(config, core, str(args.arm), int(args.outer_fold))


if __name__ == "__main__":
    main()
