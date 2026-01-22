python -m scripts.kona.__main__

# Simulation-based diagnostic probe for neural drift sources using alignment behavior

This repo is a small testbed for diagnosing **which drift sources are distinguishable** from **alignment behavior alone**.

## Problem

In long-term neural recording / BCI settings, decoding degrades over days due to drift (electrode shift, neural reorganization, timing shifts, channel mixing, dropout, etc.). A practical question is: **from the way different alignment methods help or fail, can we infer what kind of drift is present?**

This project tests that question in a controlled simulation.

## What this is

A fully self-contained pipeline:

1) **Simulate** multi-day neural-like data with controllable drift parameters  
2) Project each day into a shared latent space (day-0 PCA)  
3) Compute per-day alignment error under several alignment strategies:
- raw (no correction)
- lag (time shift)
- proc (orthogonal Procrustes)
- lag+proc
- ridge_direct (linear map fit on trials)

4) Convert those per-day errors into a compact **feature vector**  
5) Train a simple **drift-type classifier** to predict the dominant drift source:
`rotation | shear | lag | mix | dropout`

The output is cross-validated accuracy + confusion matrices, and an ablation comparing features **with vs without extra diagnostics**.

## What this is not

- Not a decoder for real neural data
- Not a claim that these simulated drift classes correspond cleanly to biology
- Not production-ready; this is a **diagnostic probe / sanity testbed**
- Not a method that “solves drift”; it is designed to expose which cases are intrinsically ambiguous given coarse alignment-only metrics

## Reproducible run

From the repo root:

```bash
source ./.venv/bin/activate
python -m scripts.kona.__main__

Cross-validated drift-type classification accuracy using alignment-behavior features. 
“Diagnostics OFF” uses only per-day alignment errors (raw/lag/proc/lag+proc/ridge). 
“Diagnostics ON” additionally includes lag magnitude, Procrustes train residual, 
    ridge-map norm, ridge train-fit error, and ridge gain over raw, which are intended to 
    disambiguate alignment failure modes.


Diagnostics OFF:
mean_acc: [0.645]
std_acc: [0.07648529270389177]

Diagnostics ON: 
mean_acc: [0.655]
std_acc: [0.07314369419163894]
