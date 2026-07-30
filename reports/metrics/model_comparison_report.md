# Validation model comparison

Models were selected without reading test targets or test predictions.
The V1 reference uses the latest weekly snapshot; tabular candidates use aggregated history; GRU uses the raw masked weekly sequence.

## Frozen selection

- **gpa**: `catboost` using `aggregated_history`; rmse=0.7277.
- **outcome**: `gru` using `raw_sequence`; f1_enroll=0.8259.
- **pace**: `xgboost` using `aggregated_history`; macro_f1=0.7692.

Operational measurements use one validation history after 10 warm-up calls and report the median of 100 calls on this machine.
