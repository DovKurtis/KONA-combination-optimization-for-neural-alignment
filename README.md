# KONA-combination-optimization-for-neural-alignment
BCI-data Python library for comparing online and offline alignment methods across session chunks or day-split datasets 
# Neural Drift Sandbox

Simulation and evaluation framework for studying neural decoder drift and simple alignment methods.

## What this does
- Simulates trial-based neural signals with structured drift
- Trains a ridge decoder
- Tests online alignment methods (`mean`, `scale`, `moment`)
- Computes trial drift vs decoder capture metrics
- Supports full-session and day-chunked evaluation

## Main functions

make_one_dataset()  
→ generate a full synthetic session

make_day_dataset()  
→ split a long session into day chunks

eval_drift(data)  
→ evaluate alignment methods on a single session

eval_dataset(days_ds)  
→ run drift evaluation across multiple days

## Example

```python
days_ds = make_day_dataset(seed=0, samples_per_day=800)
results = eval_dataset(days_ds, lambda_ridge=1.0, window_sizes=[20,50,100,200])
