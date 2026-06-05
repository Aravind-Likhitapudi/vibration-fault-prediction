"""
fault_reference.py
==================
Single source of truth for all 9 fault signatures from the reference Excel.
Each fault has 4 harmonic rows (1X, 2X, 3X, 4X+).
Each row has (H_lo, H_hi, V_lo, V_hi, A_lo, A_hi) in mm/sec.
"""

HARMONICS = ["1X", "2X", "3X", "4X+"]

FEATURE_COLS = [
    "H_1X", "V_1X", "A_1X",
    "H_2X", "V_2X", "A_2X",
    "H_3X", "V_3X", "A_3X",
    "H_4X", "V_4X", "A_4X",
]

# (H_lo, H_hi, V_lo, V_hi, A_lo, A_hi) per harmonic row
FAULT_RANGES = {
    "Angular Misalignment": [
        (0.6, 0.8,  1.0, 1.0,  2.0, 4.0),
        (0.2, 0.5,  0.3, 0.6,  0.6, 1.5),
        (0.1, 0.2,  0.1, 0.3,  0.1, 0.4),
        (0.0, 0.1,  0.0, 0.1,  0.0, 0.1),
    ],
    "Parallel Misalignment": [
        (0.6, 0.8,  1.0, 1.0,  0.2, 0.4),
        (0.7, 1.5,  1.0, 2.0,  0.1, 0.3),
        (0.1, 0.4,  0.2, 0.5,  0.0, 0.1),
        (0.0, 0.1,  0.0, 0.1,  0.0, 0.1),
    ],
    "Combined Misalignment": [
        (0.8, 1.5,  1.0, 1.8,  1.5, 3.0),
        (1.0, 2.0,  1.2, 2.5,  0.8, 1.8),
        (0.3, 0.8,  0.3, 0.8,  0.4, 1.0),
        (0.1, 0.3,  0.1, 0.3,  0.1, 0.4),
    ],
    "Unbalance": [
        (0.8, 1.5,  1.0, 1.0,  0.1, 0.3),
        (1.0, 2.0,  0.05,0.2,  0.0, 0.1),
        (0.3, 0.8,  0.0, 0.1,  0.0, 0.1),
        (0.1, 0.3,  0.0, 0.1,  0.0, 0.1),
    ],
    "Eccentricity": [
        (0.8, 1.2,  1.0, 1.0,  0.1, 0.4),
        (0.2, 0.8,  0.2, 0.8,  0.1, 0.3),
        (0.05,0.3,  0.05,0.3,  0.0, 0.1),
        (0.0, 0.1,  0.0, 0.1,  0.0, 0.1),
    ],
    "Bent Shaft": [
        (0.8, 1.2,  1.0, 1.0,  0.5, 1.5),
        (0.3, 1.0,  0.3, 1.0,  0.2, 0.8),
        (0.1, 0.4,  0.1, 0.4,  0.1, 0.3),
        (0.0, 0.2,  0.0, 0.2,  0.0, 0.2),
    ],
    "Structural Looseness": [
        (0.8, 1.2,  1.0, 1.0,  0.2, 0.6),
        (0.5, 1.0,  0.5, 1.2,  0.2, 0.5),
        (0.3, 0.8,  0.3, 0.8,  0.1, 0.4),
        (0.2, 1.0,  0.2, 1.0,  0.1, 0.5),
    ],
    "Rotating Looseness": [
        (0.8, 1.2,  1.0, 1.0,  0.3, 0.8),
        (0.6, 1.2,  0.6, 1.2,  0.2, 0.6),
        (0.5, 1.0,  0.5, 1.0,  0.2, 0.5),
        (0.5, 2.0,  0.5, 2.0,  0.2, 1.0),
    ],
    "Soft Foot": [
        (0.8, 1.2,  1.0, 1.0,  0.4, 1.0),
        (0.4, 1.0,  0.5, 1.2,  0.3, 0.8),
        (0.1, 0.5,  0.1, 0.5,  0.1, 0.4),
        (0.1, 0.3,  0.1, 0.3,  0.1, 0.2),
    ],
}

FAULT_INSIGHTS = {
    "Angular Misalignment":  "High axial (A) at 1X is 3-4x radial. Strong 2X axial. Classic coupling angle fault.",
    "Parallel Misalignment": "Strong 2X in H and V. 2X radial peak > 1X. Low axial throughout.",
    "Combined Misalignment": "Elevated energy across ALL harmonics and ALL axes simultaneously.",
    "Unbalance":             "Dominant 1X horizontal. Very low axial. Classic rotating mass fault.",
    "Eccentricity":          "Similar H and V at 1X/2X. Low axial. H ≈ V distinguishes from unbalance.",
    "Bent Shaft":            "Moderate axial at 1X with balanced H and V. Strong 2X in all axes.",
    "Structural Looseness":  "High 4X+ energy (up to 1.0 mm/sec). Many harmonics excited.",
    "Rotating Looseness":    "Highest 4X+ of all faults (up to 2.0 mm/sec). Broadband harmonic content.",
    "Soft Foot":             "Mimics misalignment. Elevated axial 1X + strong 2X. Changes with bolt torque.",
}


def range_score(val: float, lo: float, hi: float) -> float:
    """Score 0-1 for how well val fits inside [lo, hi]."""
    if lo == hi == 0:
        lo, hi = 0.0, 0.1
    if lo <= val <= hi:
        return 1.0
    if val < lo:
        return max(0.0, 1.0 - (lo - val) / (lo + 0.01))
    return max(0.0, 1.0 - (val - hi) / (hi + 0.01))


def rule_based_predict(values: list[float]) -> list[tuple[str, float]]:
    """
    values: [H_1X, V_1X, A_1X, H_2X, V_2X, A_2X, H_3X, V_3X, A_3X, H_4X, V_4X, A_4X]
    Returns: sorted list of (fault_name, score_pct) descending.
    """
    scores = {}
    for fault, rows in FAULT_RANGES.items():
        cell_scores = []
        for i, (hl, hh, vl, vh, al, ah) in enumerate(rows):
            base = i * 3
            cell_scores.append(range_score(values[base],   hl, hh))
            cell_scores.append(range_score(values[base+1], vl, vh))
            cell_scores.append(range_score(values[base+2], al, ah))
        scores[fault] = round(sum(cell_scores) / len(cell_scores) * 100, 1)
    return sorted(scores.items(), key=lambda x: -x[1])
