import json
import math
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "figures_addon"
CONFIG_PATH = ROOT / "config/analysis_config_v3.yaml"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = ROOT / "config/analysis_config.yaml"
RUN_ROOT = Path(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["paths"]["run_root"])


def _walk_json(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)
    else:
        yield None, value


def test_release_contains_no_row_level_artifacts_or_identifiers():
    if not RESULTS.exists():
        return
    for path in list(RESULTS.rglob("*")) + list(FIGURES.rglob("*")):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in {".csv", ".tsv", ".npy", ".npz", ".parquet"}, path
        if path.suffix.lower() != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in _walk_json(payload):
            if key is not None:
                lowered = key.lower()
                assert lowered not in {"record_id", "source_id", "peptide_id", "sequence", "sequences"}, (path, key)
            if isinstance(value, float):
                assert math.isfinite(value), (path, value)
            if isinstance(value, str):
                assert not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{50,}", value), path
        assert payload.get("contains_row_level_data") is False, path
        assert payload.get("contains_sequences") is False, path
        assert payload.get("contains_record_ids") is False, path


def test_private_denylist_does_not_match_tracked_addon():
    denylist = RUN_ROOT / "privacy/private_denylist.txt"
    if not denylist.is_file():
        return
    tracked = ROOT / "results"
    if not tracked.exists():
        return
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in tracked.rglob("*") if path.is_file())
    for token in denylist.read_text(encoding="utf-8").splitlines():
        if token.strip():
            assert token.strip().lower() not in text.lower()
