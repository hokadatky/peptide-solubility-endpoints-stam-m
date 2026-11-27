"""Compute one frozen V4 checkpoint for operational wall-time parallelism.

This worker does not define a second model path.  It imports the frozen V4
selection, fitting, prediction, split, contract, and atomic checkpoint
functions and applies the same loop body to exactly one independent task.
The canonical runner subsequently reloads and validates every artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper/peptide_solubility_review"
sys.path.insert(0, str(PAPER))

from analysis_addon.src.negative_controls import (  # noqa: E402
    _contract as negative_contract,
    _hash_values,
    _read_checkpoint as read_negative_checkpoint,
    _write_checkpoint as write_negative_checkpoint,
)
from analysis_addon.src.nested_models import fit_head, predict_head, select_hyperparameter  # noqa: E402
from analysis_addon.src.resumable_models import (  # noqa: E402
    _array_hash,
    _contract_json,
    _load_checkpoint,
    _save_checkpoint,
)
from analysis_addon.src.run_soluprot_reanalysis import assemble, load_config  # noqa: E402
from analysis_addon.src.split_contracts import (  # noqa: E402
    assert_no_group_overlap,
    balanced_group_folds,
    fold_hash,
    indices_for_fold,
)


CONFIG = PAPER / "analysis_addon/config/analysis_config_v4.yaml"
RUN_DIR = ROOT / "paper/general_peptide_solubility_scoping/analysis_addon_run_20260810_v4"


def primary_arm_spec(data: dict[str, object], n: int, arm: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    blocks = {
        "B0_TRANSPARENT_LOGIT": (np.asarray(data["b0"][:n], dtype=np.float64), None, "none"),
        "B1_ORDER_LOGIT": (np.asarray(data["b1"][:n], dtype=np.float64), None, "none"),
        "B0_PLUS_PROTT5_PLS_LOGIT": (
            np.asarray(data["b0"][:n], dtype=np.float64),
            np.asarray(data["embeddings"][:n], dtype=np.float64),
            "pls",
        ),
        "PROTT5_PLS_ONLY_LOGIT": (
            np.zeros((n, 0), dtype=np.float64),
            np.asarray(data["embeddings"][:n], dtype=np.float64),
            "pls",
        ),
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": (
            np.asarray(data["b1"][:n], dtype=np.float64),
            np.asarray(data["embeddings"][:n], dtype=np.float64),
            "pls",
        ),
    }
    if arm not in blocks:
        raise ValueError(f"Unknown primary arm: {arm}")
    return blocks[arm]


def compute_nested_fold(
    *,
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    mode: str,
    c_grid: list[float],
    k_grid: list[int],
    inner_folds: int,
    seed: int,
    checkpoint: Path,
    checkpoint_name: str,
    contract_tag: str,
    outer_fold: int,
) -> None:
    base = np.asarray(base, dtype=np.float64)
    embedding = None if embedding is None else np.asarray(embedding, dtype=np.float64)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    outer_folds = np.asarray(outer_folds).astype(int)
    train_rows, test_rows = indices_for_fold(outer_folds, int(outer_fold))
    assert_no_group_overlap(groups[train_rows], groups[test_rows])
    inner_assignment = balanced_group_folds(
        y[train_rows], groups[train_rows], inner_folds, int(seed) + int(outer_fold)
    )
    inner_assignment_hash = fold_hash(groups[train_rows], inner_assignment)
    global_hashes: dict[str, str | None] = {
        "base": _array_hash(base),
        "embedding": None if embedding is None else _array_hash(embedding),
        "labels": _array_hash(np.asarray(y, dtype=np.int8)),
        "groups": _array_hash(np.asarray(groups).astype("U")),
        "outer_assignment": _array_hash(np.asarray(outer_folds, dtype=np.int64)),
    }
    contract = _contract_json(
        contract_tag=contract_tag,
        global_hashes=global_hashes,
        n_rows=len(y),
        base_shape=base.shape,
        embedding_shape=None if embedding is None else embedding.shape,
        outer_fold_hash=fold_hash(groups, outer_folds),
        train_indices=train_rows,
        test_indices=test_rows,
        inner_assignment_hash=inner_assignment_hash,
        outer_fold=int(outer_fold),
        mode=mode,
        c_grid=c_grid,
        pls_k_grid=k_grid,
        inner_folds=inner_folds,
        seed=seed,
    )
    if checkpoint.is_file():
        _load_checkpoint(checkpoint, contract, test_rows)
        print(f"[parallel-worker] validated existing {checkpoint}", flush=True)
        return
    started = time.perf_counter()
    selected, grid_scores, observed_inner_hash = select_hyperparameter(
        base[train_rows],
        embedding[train_rows] if embedding is not None else None,
        y[train_rows],
        groups[train_rows],
        mode,
        c_grid,
        k_grid,
        inner_folds,
        int(seed) + int(outer_fold),
        inner_assignment=inner_assignment,
    )
    if observed_inner_hash != inner_assignment_hash:
        raise RuntimeError("Selected-model inner assignment hash changed after sealing.")
    head = fit_head(
        base[train_rows],
        embedding[train_rows] if embedding is not None else None,
        y[train_rows],
        mode,
        float(selected[1]),
        selected[0],
    )
    fold_score = predict_head(
        head,
        base[test_rows],
        embedding[test_rows] if embedding is not None else None,
    )
    row = {
        "outer_fold": int(outer_fold),
        "n_train": int(len(train_rows)),
        "n_test": int(len(test_rows)),
        "selected_pls_k": selected[0],
        "selected_c": float(selected[1]),
        "inner_fold_hash": observed_inner_hash,
        "inner_grid": grid_scores,
        "outer_fit_diagnostics": head.diagnostics,
        "fit_seconds": float(time.perf_counter() - started),
    }
    _save_checkpoint(checkpoint, contract, test_rows, fold_score, row)
    print(
        json.dumps(
            {
                "status": "created",
                "checkpoint": str(checkpoint),
                "outer_fold": int(outer_fold),
                "selected_pls_k": selected[0],
                "selected_c": float(selected[1]),
                "fit_seconds": row["fit_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def load_primary_info(run_dir: Path, arm: str) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "checkpoints/internal_primary" / arm).glob(f"{arm}_outer*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            rows.append(json.loads(str(payload["fold_info_json"].item())))
    if [int(row["outer_fold"]) for row in rows] != list(range(7)):
        raise RuntimeError(f"Primary checkpoint coverage incomplete for {arm}.")
    return rows


def compute_negative_repeat(
    *,
    data: dict[str, object],
    config: dict,
    repeat: int,
    run_dir: Path,
) -> None:
    train = data["train"]
    n = len(train)
    y = train["label"].to_numpy().astype(int)
    groups = train["cluster_id"].astype(str).to_numpy()
    folds = np.asarray(data["primary_folds"]).astype(int)
    base = np.asarray(data["b1"][:n], dtype=np.float64)
    embedding = np.asarray(data["embeddings"][:n], dtype=np.float64)
    reference_info = load_primary_info(run_dir, "B1_ORDER_LOGIT")
    candidate_info = load_primary_info(run_dir, "B1_ORDER_PLUS_PROTT5_PLS_LOGIT")
    seed = int(config["soluprot"]["label_permutation_seed"])
    contract_tag = f"{config['run_id']}:internal-primary:training-label-permutation"
    checkpoint = run_dir / "checkpoints/negative_control" / f"label_permutation_{repeat:03d}.npz"
    contract = negative_contract(
        contract_tag,
        y,
        groups,
        folds,
        reference_info,
        candidate_info,
        int(repeat),
        seed,
        _hash_values(base),
        _hash_values(embedding),
    )
    if checkpoint.is_file():
        read_negative_checkpoint(checkpoint, contract, len(y))
        print(f"[parallel-worker] validated existing {checkpoint}", flush=True)
        return
    ref_by_fold = {int(row["outer_fold"]): row for row in reference_info}
    cand_by_fold = {int(row["outer_fold"]): row for row in candidate_info}
    reference = np.full(n, np.nan, dtype=float)
    candidate = np.full(n, np.nan, dtype=float)
    started = time.perf_counter()
    for outer_fold in sorted(np.unique(folds).tolist()):
        train_rows, test_rows = indices_for_fold(folds, int(outer_fold))
        assert_no_group_overlap(groups[train_rows], groups[test_rows])
        rng = np.random.default_rng(seed + int(repeat) * 1009 + int(outer_fold))
        permuted_y = rng.permutation(y[train_rows])
        ref_row = ref_by_fold[int(outer_fold)]
        cand_row = cand_by_fold[int(outer_fold)]
        ref_head = fit_head(
            base[train_rows], None, permuted_y, "none", float(ref_row["selected_c"]), None
        )
        cand_head = fit_head(
            base[train_rows],
            embedding[train_rows],
            permuted_y,
            "pls",
            float(cand_row["selected_c"]),
            int(cand_row["selected_pls_k"]),
        )
        reference[test_rows] = predict_head(ref_head, base[test_rows], None)
        candidate[test_rows] = predict_head(cand_head, base[test_rows], embedding[test_rows])
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise RuntimeError("Negative-control OOF prediction is incomplete.")
    write_negative_checkpoint(checkpoint, contract, reference, candidate)
    print(
        json.dumps(
            {
                "status": "created",
                "checkpoint": str(checkpoint),
                "repeat": int(repeat),
                "fit_seconds": float(time.perf_counter() - started),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["primary", "stability", "negative-control"], required=True)
    parser.add_argument("--arm")
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--stability-seed", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(CONFIG)
    data = assemble(config)
    run_dir = RUN_DIR
    if args.task == "negative-control":
        if args.repeat is None or not 0 <= args.repeat < int(config["soluprot"]["label_permutation_replicates"]):
            raise ValueError("A valid --repeat is required.")
        compute_negative_repeat(data=data, config=config, repeat=args.repeat, run_dir=run_dir)
        return

    if args.outer_fold is None or not 0 <= args.outer_fold < int(config["soluprot"]["primary_outer_folds"]):
        raise ValueError("A valid --outer-fold is required.")
    train = data["train"]
    n = len(train)
    y = train["label"].to_numpy().astype(int)
    groups = train["cluster_id"].astype(str).to_numpy()
    c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
    k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
    inner_folds = int(config["soluprot"]["inner_folds"])

    if args.task == "primary":
        if args.arm is None:
            raise ValueError("--arm is required for primary tasks.")
        base, embedding, mode = primary_arm_spec(data, n, args.arm)
        outer_folds = np.asarray(data["primary_folds"]).astype(int)
        seed = int(config["global_seed"])
        target_dir = args.output_dir or run_dir / "checkpoints/internal_primary" / args.arm
        checkpoint = target_dir / f"{args.arm}_outer{args.outer_fold:02d}.npz"
        contract_tag = f"{config['run_id']}:internal-primary:{args.arm}"
        checkpoint_name = args.arm
    else:
        if args.stability_seed not in [int(value) for value in config["soluprot"]["fold_stability_seeds"]]:
            raise ValueError("A frozen --stability-seed is required.")
        if args.arm not in {"reference", "candidate"}:
            raise ValueError("Stability --arm must be reference or candidate.")
        base = np.asarray(data["b1"][:n], dtype=np.float64)
        if args.arm == "reference":
            embedding = None
            mode = "none"
            checkpoint_name = "B1_ORDER_LOGIT"
        else:
            embedding = np.asarray(data["embeddings"][:n], dtype=np.float64)
            mode = "pls"
            checkpoint_name = "B1_ORDER_PLUS_PROTT5_PLS_LOGIT"
        outer_folds = balanced_group_folds(
            y,
            groups,
            int(config["soluprot"]["primary_outer_folds"]),
            int(args.stability_seed),
            min_class_groups_per_fold=20,
        )
        seed = int(args.stability_seed)
        target_dir = args.output_dir or (
            run_dir / "checkpoints/fold_stability" / f"seed_{seed}" / args.arm
        )
        checkpoint = target_dir / f"{checkpoint_name}_outer{args.outer_fold:02d}.npz"
        contract_tag = f"{config['run_id']}:stability:{seed}:{args.arm}"

    compute_nested_fold(
        base=base,
        embedding=embedding,
        y=y,
        groups=groups,
        outer_folds=outer_folds,
        mode=mode,
        c_grid=c_grid,
        k_grid=k_grid,
        inner_folds=inner_folds,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_name=checkpoint_name,
        contract_tag=contract_tag,
        outer_fold=int(args.outer_fold),
    )


if __name__ == "__main__":
    main()
