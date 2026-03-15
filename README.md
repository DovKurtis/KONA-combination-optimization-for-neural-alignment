# KONA-combination-optimization-for-neural-alignment
BCI-data Python library for comparing online and offline alignment methods across session chunks or day-split datasets 

This project simulates drifting trial-related neural signals, trains a ridge decoder, and evaluates simple alignment methods (`mean`, `scale`, `moment`, `meanstatic`) under drift.

It supports:
- full-session evaluation
- day-chunked evaluation
- trial-level drift vs capture plots
- window sweeps for online correction methods

Main entry points:
- `make_one_dataset(...)`
- `make_day_dataset(...)`
- `eval_drift(...)`
- `eval_dataset(...)`

Example:
days_ds = make_day_dataset(seed=0, samples_per_day=800)
results = eval_dataset(days_ds, lambda_ridge=1.0, window_sizes=[20, 50, 100, 200])
