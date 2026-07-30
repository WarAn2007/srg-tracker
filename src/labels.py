"""Pure target-construction functions for SRG-Tracker."""

from __future__ import annotations

from numbers import Real


def _linear(
    value: float,
    input_low: float,
    input_high: float,
    output_low: float,
    output_high: float,
) -> float:
    ratio = (value - input_low) / (input_high - input_low)
    return output_low + ratio * (output_high - output_low)


def score_to_gpa(score: Real) -> float:
    """Convert a final 0-100 score to the approved 0.00-4.50 GPA scale."""
    value = float(score)
    if not 0.0 <= value <= 100.0:
        raise ValueError("final_course_score must be between 0 and 100.")
    if value < 65.0:
        result = 0.0
    elif value <= 70.0:
        result = _linear(value, 65.0, 70.0, 2.00, 2.49)
    elif value <= 75.0:
        result = _linear(value, 70.0, 75.0, 2.50, 2.99)
    elif value <= 80.0:
        result = _linear(value, 75.0, 80.0, 3.00, 3.49)
    elif value <= 90.0:
        result = _linear(value, 80.0, 90.0, 3.50, 3.99)
    elif value <= 95.0:
        result = _linear(value, 90.0, 95.0, 4.00, 4.49)
    else:
        result = 4.50
    return round(min(max(result, 0.0), 4.50), 2)
