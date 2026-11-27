"""Exact STAM descriptor function extraction; no fitted app model imported."""
import math
import numpy as np
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
AA_ALPHABET="ACDEFGHIKLMNPQRSTVWY"
KD_SCALE=ProtParamData.kd

def _sequence_entropy(fractions: np.ndarray) -> float:
    positive = fractions[fractions > 0]
    if positive.size <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(len(AA_ALPHABET)))

def _maximum_window_hydropathy(sequence: str, window: int = 9) -> float:
    values = np.asarray([KD_SCALE[aa] for aa in sequence], dtype=float)
    if len(values) <= window:
        return float(values.mean())
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return float(means.max())

def _aliphatic_index(fractions: dict[str, float]) -> float:
    return 100.0 * (
        fractions["A"]
        + 2.9 * fractions["V"]
        + 3.9 * (fractions["I"] + fractions["L"])
    )

def extract_sequence_features(sequence: str) -> dict[str, float]:
    """Reproduce the frozen transparent SoluProt feature definitions exactly."""

    sequence = normalize_sequence(sequence)
    analysis = ProteinAnalysis(sequence)
    counts = analysis.count_amino_acids()
    length = len(sequence)
    fractions = {aa: counts[aa] / length for aa in AA_ALPHABET}
    fraction_array = np.asarray([fractions[aa] for aa in AA_ALPHABET], dtype=float)
    charge_ph7 = float(analysis.charge_at_pH(7.0))
    helix, turn, sheet = analysis.secondary_structure_fraction()
    aromatic_fraction = float(analysis.aromaticity())
    mean_aggregation = float(
        np.mean([AGGRESCAN_PROPENSITY[residue] for residue in sequence])
    )
    aggregation_proxy = (1.0 - 0.2 * aromatic_fraction) / (
        1.0 + np.exp(5.0 * mean_aggregation)
    )

    return {
        **{f"aac_{aa}": fractions[aa] for aa in AA_ALPHABET},
        "length": float(length),
        "log1p_length": float(math.log1p(length)),
        "molecular_weight_da": float(analysis.molecular_weight()),
        "isoelectric_point": float(analysis.isoelectric_point()),
        "charge_at_ph7": charge_ph7,
        "charge_density_ph7": charge_ph7 / length,
        "instability_index": float(analysis.instability_index()),
        "normalized_sequence_entropy": _sequence_entropy(fraction_array),
        "max_kd_window9": _maximum_window_hydropathy(sequence, window=9),
        "helix_propensity_fraction": float(helix),
        "turn_propensity_fraction": float(turn),
        "sheet_propensity_fraction": float(sheet),
        "whole_sequence_gravy": float(analysis.gravy()),
        "aromaticity_fraction": aromatic_fraction,
        "aliphatic_index": _aliphatic_index(fractions),
        "mean_aggregation_propensity": mean_aggregation,
        "aggregation_solubility_proxy_0_100": float(100.0 * aggregation_proxy),
    }

import re
AA_SET=frozenset(AA_ALPHABET)
AGGRESCAN_PROPENSITY = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        [
            -0.037,
            -0.088,
            -0.065,
            -0.146,
            0.402,
            -0.097,
            -0.079,
            -0.042,
            0.102,
            0.419,
            0.321,
            -0.183,
            0.295,
            0.318,
            -0.172,
            -0.030,
            0.009,
            0.339,
            0.218,
            0.242,
        ],
    )
)

class InputValidationError(ValueError):
    """An input sequence cannot be represented by the canonical model."""

def normalize_sequence(raw: object, *, max_length: int = 5000) -> str:
    """Normalize case/whitespace and fail closed on non-canonical chemistry."""

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raise InputValidationError("配列が空です。")
    sequence = re.sub(r"\s+", "", str(raw)).upper()
    if not sequence:
        raise InputValidationError("配列が空です。")
    if len(sequence) > max_length:
        raise InputValidationError(
            f"配列長 {len(sequence)} はアプリ上限 {max_length} 残基を超えています。"
        )
    invalid = sorted(set(sequence) - AA_SET)
    if invalid:
        rendered = ", ".join(invalid)
        raise InputValidationError(
            "canonical 20アミノ酸以外を検出しました: "
            f"{rendered}。D体、N-methyl化、staple、環状化、末端修飾、X/B/Z/U/O/J、"
            "ギャップを一文字配列へ黙って変換しないでください。"
        )
    return sequence

