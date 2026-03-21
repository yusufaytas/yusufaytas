"""Scoring functions for prose quality."""


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def score_range(
    value: float,
    ideal_low: float,
    ideal_high: float,
    hard_low: float,
    hard_high: float
) -> float:
    """Score a value based on ideal and hard ranges."""
    if value < hard_low or value > hard_high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * (value - hard_low) / (ideal_low - hard_low)
    return 100.0 * (hard_high - value) / (hard_high - ideal_high)


def score_min(value: float, ideal: float, hard_low: float) -> float:
    """Score a value with minimum threshold."""
    if value <= hard_low:
        return 0.0
    if value >= ideal:
        return 100.0
    return 100.0 * (value - hard_low) / (ideal - hard_low)
