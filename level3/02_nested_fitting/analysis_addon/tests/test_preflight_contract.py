from pathlib import Path

import pytest
import yaml

from analysis_addon.src.preflight import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_v3_recovery_config_is_explicitly_accepted():
    config = load_config(ROOT / "config/analysis_config_v3.yaml")
    assert config["run_id"] == "addon_20260809_v3"
    assert config["recovery"]["deferred_arm_multiplicity_p_value"] == 1.0


def test_unknown_top_level_config_key_is_rejected(tmp_path):
    config = yaml.safe_load((ROOT / "config/analysis_config_v3.yaml").read_text(encoding="utf-8"))
    config["silent_default"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*silent_default"):
        load_config(path)
