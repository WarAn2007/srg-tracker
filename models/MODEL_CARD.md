# SRG-Tracker packaged models

The repository includes the pre-trained models required by the local web
application. No dataset generation or training is required to run the demo.

- `gpa_model.joblib`: MLP regressor for final GPA.
- `outcome_model.joblib`: linear classifier for `pass` or `enroll` outcome.
- `pace_model.joblib`: linear classifier for `behind`, `on_track`, or `ahead`.
- `selection.json`: the manifest that binds each prediction task to its model.

These files must remain together in `models/`. They are loaded by FastAPI at
prediction time.

## Intended use

This is an educational early-warning demo for instructor review. It uses synthetic
course histories and must not make automatic academic decisions or rank students.

## Inputs

Models receive only consecutive observed weekly history through the selected
cutoff. Identifiers, targets, future weeks, and completed-course outcomes are
excluded from every model feature representation.

## Limitations

- Synthetic data does not establish performance on a real institution.
- Outputs may be wrong, especially with short histories.
- The user must interpret probability and pace together with course context.
- Human review and institutional validation are required before any real use.
