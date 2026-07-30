# SRG-Tracker V2 — prompt for research and implementation

## Role and working rules

You are improving the existing SRG-Tracker ML capstone. Implement V2 as a new, reproducible version without silently breaking V1. Start by inspecting the current repository, dataset card, tests, and training/evaluation code. Then provide a short implementation plan and wait for approval before editing code.

Use English for all source code, tests, metrics, dataset cards, model cards, CSV column names, and Markdown artifacts. Explain decisions in clear Russian when communicating with the project owner.

Do not use a real or external dataset in V2. Improve the synthetic generator instead. The data remains a demonstration and must not be used for real academic decisions.

## Objective

Build a history-aware early-warning system that predicts, from all information available **up to the student's latest submitted week**:

1. `final_gpa` — regression in the inclusive range `0.00` to `4.50`;
2. `course_outcome` — binary classification: `pass` or `enroll`;
3. `learning_pace` — multiclass classification: `behind`, `on_track`, or `ahead`.

`enroll` means the student failed and must repeat the course. The pass threshold remains `final_course_score >= 65`.

The user supplies a student's weekly history for one course and semester, not a single weekly snapshot. The service must validate the supplied history, derive features available at the latest week, and return all three predictions.

## Non-negotiable anti-leakage rules

Treat each prediction time as a cutoff. A feature may use only facts dated on or before that cutoff.

- Never use `student_id` as a model feature.
- Never use `final_course_score`, `final_gpa`, `course_outcome`, `learning_pace`, or data calculated from any of them as model inputs.
- Never use `final_exam_score_audit_only` as a model input, history feature, imputation value, scaler-fit value, or training target proxy.
- `final_exam_score_audit_only` is an audit-only course-completion field. It contributes 25% to the generated `final_course_score`, but is unavailable before the final exam and excluded from every prediction model.
- `midterm_score` is unavailable before the midterm checkpoint. It may be used only from week 7 onward and only if the midterm was actually recorded.
- Do not fill a missing past value with a future value, course final average, final score, or target-derived statistic.
- Split by unique `student_id` before any fit, encoding, normalization, feature selection, hyperparameter search, or resampling. Train/validation/test students must remain disjoint.
- Preserve the untouched V2 test split. Select models only using train and validation data; evaluate once on the test split after the selection is frozen.

Add focused automated tests for every rule above.

## GPA definition

Generate `final_gpa` deterministically from `final_course_score = x`, then round to two decimal places. Use the following piecewise rule. Within a band, use linear interpolation; cap the upper endpoint at the listed maximum.

| Final score `x` | GPA rule |
| --- | --- |
| `x > 95` | `4.50` |
| `90 < x <= 95` | linear from `4.00` (just above 90) to `4.49` (at 95) |
| `80 < x <= 90` | linear from `3.50` (just above 80) to `3.99` (at 90) |
| `75 < x <= 80` | linear from `3.00` (just above 75) to `3.49` (at 80) |
| `70 < x <= 75` | linear from `2.50` (just above 70) to `2.99` (at 75) |
| `65 <= x <= 70` | linear from `2.00` (at 65) to `2.49` (at 70) |
| `x < 65` | `0.00` |

Implement this as one tested pure function. Boundary tests must explicitly cover `64.99`, `65`, `70`, `75`, `80`, `90`, `90.01`, `95`, and `95.01`. In particular, `x = 90` must yield `3.99`, and `x > 95` must yield `4.50`.

## Revised final-score generator

Create final scores only to generate labels and audit the synthetic data. They are not features.

Use this corrected, transparent 100% formula:

```text
final_course_score =
    0.25 * final_exam_score_audit_only
  + 0.25 * midterm_score
  + 0.10 * attendance_component
  + 0.10 * average_weekly_grade
  + 0.15 * average_assignment_score
  + 0.15 * average_quiz_score
```

The weights must sum exactly to `1.00`; test this invariant. Explain in the dataset card that the final exam has the required 25% weight and that the prior V1 formula totalled 1.10. Clip generated final scores to 0–100 only after the formula and noise are applied.

## Realistic synthetic data requirements

Keep the existing unit of observation: one student-course-semester-week record, with a weekly record for every enrolled student. Keep a reproducible seed and write all generation assumptions to the dataset card.

Replace overly deterministic relationships with a coherent latent-data process:

- Draw stable student traits such as preparation, engagement, consistency, and punctuality.
- Draw course and semester effects such as difficulty, workload, and assessment strictness.
- Generate weekly grades, attendance, submission delays, quiz results, and assignment results from those factors plus week-level noise and mild temporal dependence.
- Allow realistic improvement, decline, and occasional disruption; do not make the final grade a near-direct copy of one earlier measure.
- Ensure attendance, lateness, grades, assignments, and quizzes are correlated in plausible directions, but not perfectly correlated.
- Make assessment availability realistic: a student must have at least one assignment and one quiz before the midterm, and at least one assignment and one quiz before final audit data is recorded.
- Generate `midterm_score` at week 7. It is missing/unavailable before week 7 and then remains fixed for that course attempt.
- Generate `final_exam_score_audit_only` only at course completion. It is absent from early-week prediction data and present only in audit/full-course records.
- Validate ranges, label distributions, missingness rules, inter-feature correlations, trend examples, weekly record counts, and student-disjoint splits. Save a concise data-quality report.

Do not add demographic attributes or personally identifying data.

## Prediction input contract and validation

Implement a documented input schema for one student's ordered course history. Required identity/context fields are `course_id`, `semester`, and a consistent course attempt identifier that is not passed to the model. Weekly records must include their week number and any observations recorded that week.

Validation must:

1. Reject an empty history, duplicate week numbers, unordered or non-consecutive weeks, week numbers outside the configured course duration, values outside valid ranges, mixed courses/semesters, and attempts to submit future-only fields early.
2. Identify the cutoff as the maximum supplied week.
3. Require at least one weekly record before prediction.
4. Before week 7, permit a missing `midterm_score`; at and after week 7, require it. Reject a midterm score supplied before week 7.
5. Require at least one completed assignment and one completed quiz before the midterm checkpoint. Enforce the same minimum before final audit data is recorded.
6. Treat `final_exam_score_audit_only` as audit-only: reject it in ordinary early-warning prediction requests. If a separate full-course audit payload supports it, validate it but remove it before any feature construction/model call.
7. Return clear field-level validation errors; never silently invent missing observations.

Document the exact JSON-like payload shape and include valid examples for weeks 4, 7, and 14 plus invalid examples for missing midterm and leaked final-exam data.

## Two history representations

Build and compare both representations using the same cutoff dates, same students, and same target labels.

### A. Aggregated historical features for tabular models

For each cutoff, create one row with only past/current information. Include:

- cutoff week and number of observed weeks;
- current value and cumulative mean for weekly grades, attendance, assignment scores, quiz scores, delays, corrections, completions;
- median, minimum, maximum, standard deviation, and last-3-week mean where meaningful;
- trend/slope for weekly grade and attendance when at least two weeks exist;
- changes from the first observed week and rolling volatility;
- cumulative assignment and quiz completion rates;
- `midterm_score_available` and `midterm_score` only when allowed by the cutoff;
- categorical course and semester context encoded inside each training pipeline.

Features must be deterministic, unit-tested, and named clearly. A history ending at week `k` may never incorporate row `k + 1` or later.

### B. Raw weekly sequence for GRU

Feed weekly history directly to a GRU rather than first reducing it to aggregates.

- Create one training example per student-course-semester-cutoff with tensor shape `[sequence_length, weekly_feature_count]`.
- Use only non-leaking weekly fields. Do not include any final target, final score, or final-exam audit field.
- Pad shorter histories consistently and use masking so padding does not become evidence.
- Encode categorical context safely; keep categorical transformations fit on train data only.
- Represent unavailable midterm information with an explicit availability indicator plus a safe missing-value representation. Do not confuse absence with a zero score.
- Use a shared GRU encoder with three task heads (GPA regression, outcome classification, pace classification), or three clearly justified separate GRUs. State the choice and parameter count.
- Use Adam, deterministic seeds where supported, early stopping based on validation metrics, and checkpoint the best validation model.

## Models and fair experiment design

Retain the current V1 HistGradientBoosting result as the initial reference. Train the following V2 candidates:

1. Simple baselines: Ridge regression for GPA and Logistic Regression for classifications.
2. HistGradientBoosting on aggregated historical features.
3. XGBoost on aggregated historical features.
4. CatBoost on aggregated historical features, including course/semester categorical handling where appropriate.
5. MLP with Adam on aggregated historical features.
6. GRU with Adam on raw weekly sequences.

Use the same V2 student-disjoint split, target construction, cutoff protocol, random seeds, and train/validation tuning budget for all candidates. Add a dependency only when it is necessary and document its compatible version. If XGBoost, CatBoost, or the deep-learning framework is unavailable, report the exact installation command and stop for approval rather than substituting a different model silently.

Run an ablation that compares: (a) the latest-week row only, (b) aggregated history, and (c) GRU sequence. This is the core V2 research question.

Do not claim that a model is better merely from training performance. Use validation for selection and untouched test performance for the final comparison.

## Metrics, reports, and selection criteria

Report every model by task and cutoff week (at least weeks 4, 7, 10, and 14):

| Task | Required metrics |
| --- | --- |
| GPA regression | MAE, RMSE, R², fraction within ±0.25 GPA |
| Course outcome | `enroll` recall, `enroll` F1, ROC-AUC, confusion matrix |
| Learning pace | macro-F1, balanced accuracy, class-wise recall, confusion matrix |
| Operational cost | train time, median single-history inference time, serialized model size, parameter count for neural models |

Also report calibration for the classification probabilities and an error breakdown by cutoff week and course. Measure all times on the same machine and state the hardware/software environment. Do not compare inference times without using equivalent input sizes and warm-up conditions.

Choose a production candidate only if it improves the relevant validation metric over the V1 reference or offers a clearly documented operational advantage. Prefer the simpler model when results are practically tied. Freeze the choice before the one-time test evaluation.

## Deliverables

Create V2-specific code and artifacts with names that do not overwrite V1 assets until V2 validation passes. Adapt filenames to the existing repository layout after inspection.

- updated synthetic generator and deterministic V2 processed splits;
- GPA conversion module plus boundary tests;
- history-input validation and feature/sequence builders plus anti-leakage tests;
- training entry points for all listed models;
- saved V2 models and inference API that consumes a full history;
- data card describing the new synthetic mechanism and limitations;
- experiment configuration with seeds and model hyperparameters;
- validation and final held-out test metrics in CSV/Markdown;
- plots for performance by cutoff, calibration, and observed-vs-predicted GPA;
- concise model-comparison report recommending one model per task or one justified shared approach;
- README update with reproducible PowerShell commands.

## Definition of done

V2 is complete only when all of the following are true:

1. GPA boundaries and score-weight sum tests pass.
2. No prediction feature contains future information, final targets, or `final_exam_score_audit_only`.
3. Every split is student-disjoint and the test split was not used during model selection.
4. The input validator enforces the week-7 midterm and assignment/quiz rules with readable errors.
5. The GRU receives raw chronological weekly sequences with masking, not pre-aggregated rows.
6. V1, tabular-history models, and GRU are compared under the same protocol.
7. Results include predictive metrics, runtime, model size, uncertainty/calibration evidence, and limitations.
8. Generation, training, final evaluation, and the complete automated test suite run successfully from a clean environment using documented commands.

## Execution order

1. Inspect V1 and summarize the change plan; wait for approval.
2. Define schemas, GPA function, revised score formula, and tests.
3. Build and validate the realistic V2 synthetic dataset and student-disjoint splits.
4. Implement history validation plus aggregated and sequence feature builders.
5. Establish simple and V1-reference baselines.
6. Train/tune tabular models, MLP, and GRU only on train/validation data.
7. Run ablations and choose candidates from validation evidence.
8. Run the single final test evaluation, write reports, update documentation, and present limitations.
