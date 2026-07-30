# SRG-Tracker V2.2 model card

## Frozen selection

- GPA: CatBoost on aggregated history.
- Course outcome: shared GRU outcome head on raw masked weekly sequences.
- Learning pace: XGBoost on aggregated history.

Selection used validation only. `selection.json` was written before the held-out
test evaluation.

## Inputs and outputs

The public API consumes one ordered student-course-attempt history through a
cutoff and returns GPA, pass/enroll, enroll probability, learning pace, and an
advisory. Identifiers, targets, final score, and final-exam audit data are excluded
from model inputs.

## Intended use

Educational capstone demonstration, reproducibility study, leakage testing, and
comparison of snapshot, aggregated-history, and raw-sequence representations.

## Prohibited use

Real grading, enrolment, ranking, discipline, student access decisions, causal
recommendations, or deployment without institutional approval and human review.

## Limitations

All training and evaluation data is synthetic. Reported performance measures
recovery of generator patterns, not real-world educational validity or fairness.
