from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis_addon.src.run_soluprot_reanalysis import run_external, run_internal


def test_v4_reduced_internal_to_external_smoke(tmp_path):
    rng = np.random.default_rng(131)
    n_dev, n_ext = 240, 80
    y_dev = np.tile([0, 1], n_dev // 2)
    y_ext = np.tile([0, 1], n_ext // 2)
    train = pd.DataFrame({
        "record_id": [f"dev_{i:03d}" for i in range(n_dev)],
        "cluster_id": [f"devg_{i:03d}" for i in range(n_dev)],
        "label": y_dev,
        "partition": ["fit"] * 120 + ["model_selection"] * 60 + ["calibration"] * 60,
    })
    external = pd.DataFrame({
        "record_id": [f"ext_{i:03d}" for i in range(n_ext)],
        "external_cluster_id": [f"extg_{i:03d}" for i in range(n_ext)],
        "label": y_ext,
    })
    b0 = rng.normal(size=(n_dev + n_ext, 5))
    b1 = rng.normal(size=(n_dev + n_ext, 8))
    embedding = rng.normal(size=(n_dev + n_ext, 12))
    b1[:n_dev, 0] += 0.7 * y_dev
    embedding[:n_dev, 0] += y_dev
    b1[n_dev:, 0] += 0.7 * y_ext
    embedding[n_dev:, 0] += y_ext
    data = {
        "train": train,
        "external": external,
        "b0": b0,
        "b1": b1,
        "embeddings": embedding,
        "primary_folds": np.repeat(np.arange(4), n_dev // 4),
    }
    source = Path(__file__).parents[1] / "config" / "analysis_config_v4.yaml"
    config = deepcopy(yaml.safe_load(source.read_text()))
    config["run_id"] = "v4_integration_smoke"
    config["paths"]["run_root"] = str(tmp_path)
    config["statistics"]["bootstrap_replicates_final"] = 20
    config["soluprot"]["primary_outer_folds"] = 4
    config["soluprot"]["inner_folds"] = 3
    config["soluprot"]["fold_stability_seeds"] = [131]
    config["soluprot"]["label_permutation_replicates"] = 20
    config["soluprot"]["logistic_c_grid"] = [0.1, 1.0]
    config["soluprot"]["pls_k_grid"] = [1, 2]
    internal = run_internal(config, tmp_path, data)
    assert internal["status"] == "completed"
    assert len(internal["arms"]) == 6
    assert len(internal["fold_stability"]) == 1
    assert internal["negative_controls"]["training_label_permutation"]["status"] == "PASSED"
    external_result = run_external(config, tmp_path, data)
    assert external_result["status"] == "completed"
    seal = tmp_path / "manifests" / "external_development_only_seal.json"
    assert seal.is_file()
    assert (tmp_path / "raw_results" / "internal_primary.json").is_file()
    assert (tmp_path / "raw_results" / "external_retrospective.json").is_file()
