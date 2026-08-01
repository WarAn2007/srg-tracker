# SRG-Tracker V2.3 model card

## Configured production selection

- GPA: MLP with Adam on aggregated history (validation RMSE 0.7376; rank 3).
- Course outcome: logistic regression on aggregated history (enroll F1 0.8247; rank 3).
- Learning pace: logistic regression on aggregated history (macro-F1 0.7600; rank 3).

These are deliberate user-configured production choices made after validation
review, not automatic validation winners. The mapping is version-controlled in
`src/config.py` as `USER_SELECTED_MODELS`; `selection.json` records the method,
seed, metric, and validation rank before a held-out test evaluation in a new
experiment.

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
