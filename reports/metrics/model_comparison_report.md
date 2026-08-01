# Validation model comparison

Models were selected without reading test targets or test predictions.
The V1 reference uses the latest weekly snapshot; tabular candidates use aggregated history; GRU uses the raw masked weekly sequence.
V1 reference rows are included only when their frozen artifacts are available; missing artifacts are reported and do not block V2.3 training.

## Frozen configured selection

- **gpa**: `mlp_adam` using `aggregated_history`; rmse=0.7376; validation rank 3.
- **outcome**: `linear` using `aggregated_history`; f1_enroll=0.8247; validation rank 3.
- **pace**: `linear` using `aggregated_history`; macro_f1=0.7600; validation rank 3.

Operational measurements use one validation history after 10 warm-up calls and report the median of 100 calls on this machine.
