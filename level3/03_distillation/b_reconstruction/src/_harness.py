"""Shared loader for the on-device distillation reproduction.

Reuses the frozen 2026-08-13 distillation contract's data layer verbatim
(teacher_crossfit.load_development / load_prott5 / make_crossfit_plan) so the
reproduction cannot silently drift from the audited development partition.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("STAM_WORKSPACE", "."))
PLAN_SRC = PROJECT_ROOT / "paper/peptide_solubility_review_distillation_plan_20260813/src"
sys.path.insert(0, str(PLAN_SRC))

import teacher_crossfit as tc  # noqa: E402

CONFIG_PATH = (
    PROJECT_ROOT
    / "paper/peptide_solubility_review_distillation_plan_20260813/config/DISTILLATION_CONFIG_20260813.json"
)


def load():
    config = tc.load_config(CONFIG_PATH)
    tc.validate_input_hashes(config, include_esm2=False)
    data = tc.load_development(config)
    prott5 = tc.load_prott5(config)
    return config, data, prott5
