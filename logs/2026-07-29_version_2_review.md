# SRG-Tracker — Version 2 Planning Review

**Planned version:** 2.0  
**Starting point:** completed version 1.1

## Goal

Build on the first working ML version without invalidating its held-out test results. Version 2 will improve model-selection discipline, make predictions easier for non-technical users to access, and explain model outputs.

## Planned changes

### 1. Tune models using train and validation only

- Keep the current test metrics as the recorded result for version 1.1.
- Use only `train.csv` and `validation.csv` for model selection and hyperparameter tuning.
- Do not repeatedly optimise against the existing `test.csv`; this would make its result overly optimistic.
- Before a new final comparison, create a new independent test split or a new dataset version.

### 2. Add `RandomizedSearchCV` for HistGradientBoosting

- Create a reproducible parameter-search workflow for each task.
- Search meaningful parameters such as `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, and `l2_regularization`.
- Select candidates by the agreed validation metric: RMSE for score, F1 for `enroll`, and Macro-F1 for learning pace.
- Save the parameters, validation metrics, random seed, and selected model for reproducibility.

### 3. Create a clearer user interface

- Build a Streamlit form for one weekly student-course record.
- Validate required fields and numeric ranges before inference.
- Show predicted final score, `enroll` probability, course outcome, learning pace, and an advisory message in plain language.
- Include a visible limitation notice: the model is trained on synthetic data and must not make automatic academic decisions.

### 4. Add prediction explainability

- Start with global feature importance for each trained model.
- Add local explanations for an individual prediction, preferably with SHAP when the dependency and runtime are suitable.
- Express explanations as associations in the data, not as proven causes.
- Present a concise explanation in the UI, for example that low recent grades and missed work are associated with higher predicted risk.

## Proposed v2 sequence

1. Define a new data/test evaluation protocol.
2. Implement and validate randomized hyperparameter search.
3. Compare tuned candidates on validation.
4. Generate a new independent final evaluation.
5. Add explainability artifacts and tests.
6. Build and test the Streamlit interface.
7. Update README, instructions, metrics and limitations.

## Success criteria

- Every tuning decision is made without using the v1.1 test split.
- Search results are saved and reproducible.
- A non-technical user can receive a prediction through a form.
- The interface explains the prediction without claiming causal certainty.
- Automated tests cover input validation, model loading, and explanation output.
