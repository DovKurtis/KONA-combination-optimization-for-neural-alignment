OVERVIEW
--------

This repository simulates multi-day neural recordings with controlled drift and evaluates alignment methods that recover a shared latent structure across days.

The pipeline has three layers:
(1) drift presets,
(2) data generation,
(3) alignment + evaluation.

A thin run script (“cockpit”) wires these components together.


drift_presets.py
----------------

drift_presets.py defines the SimConfig object, which specifies dataset shape, noise magnitudes, and which drift mechanisms are active.

Non-preset parameters that define dataset shape and noise:

    n_days        : number of recording days
    n_trials      : trials per day
    T             : timepoints per trial
    d_latent      : latent dimensionality
    n_ch          : number of channels
    seed          : RNG seed

Noise parameters:

    obs_rho       : temporal correlation of per-channel noise
    obs_scale     : scale of per-channel noise
    cm_rho        : temporal correlation of common-mode noise
    cm_scale      : scale of common-mode noise

Preset parameters define the type of drift applied to the data:

    PURE_ROTATION        : latent rotation only
    ROTATION_PLUS_LAG    : latent rotation + temporal lag
    SHEAR_ONLY           : non-orthogonal latent shear
    MIXING_DRIFT         : slow drift in channel mixing
    CHANNEL_DROPOUT      : channel loss
    HARD_COMBO           : rotation + shear + lag + mixing drift + dropout

Changing the experiment means selecting or modifying one preset.


sim_core.py
-----------

sim_core.py generates the synthetic neural data.

It consumes a SimConfig and outputs:

    X_n_list
    shape: (cfg.n_days, cfg.n_trials, cfg.T, cfg.n_ch)

Generation proceeds as follows:

(1) Generate a single shared latent tensor:

        Z_trials
        shape: (trials, time, d_latent)

(2) Sample per-day latent transforms and per-day mixing matrices:

        R_list, S_list, lag_list
        W_list

(3) For each day:
    - apply temporal lag via np.roll on the time axis
    - apply latent drift via:
            Zt @ (S @ R)
    - mix latents into channels via:
            @ Wd
    - add colored observation noise

The output is a realistic multi-day neural recording tensor with known ground-truth structure.


align_X_n_list_and_evaluate.py
------------------------------

align_X_n_list_and_evaluate.py consumes X_n_list and evaluates alignment quality.

It first fixes a single preprocessing pipeline using day 0 only:

    w0   : channel weights
    sc0  : StandardScaler
    pca0 : PCA

That same day-0 pipeline is then used to convert every day’s raw channels into latent trajectories:

    Z_day
    shape: (trials, time, k)


run_alignment_pipeline_kfold
----------------------------

For each day > 0, this function runs K-fold cross-validation over trials and, per fold, computes:

(1) raw error between day and day-0 mean latent trajectories
(2) best temporal lag chosen by brute-force Frobenius minimization
(3) Procrustes rotation R fit on lag-aligned TRAIN means
(4) residual ridge map W_res fit on lag-aligned, rotated TRAIN trials
    to predict (day0 − day) residuals

R and W_res are applied to TEST data, and the final Frobenius error is reported.


sweep_k_lambda_fast
-------------------

This is a fast hyperparameter search.

It loops over latent dimension k and ridge regularization lambda_,
rebuilds the day-0 scaler and PCA for each k,
fits a ridge map on TRAIN trials,
scores on TEST means,
and returns the best (k, lambda_).


run_one_line
------------

run_one_line is a convenience wrapper.

It calls sweep_k_lambda_fast to select k and lambda_,
then runs run_alignment_pipeline_kfold using those values.

It should not execute automatically at import time.


cockpit / run script
--------------------

The run script contains no scientific logic.

It:
    - selects a preset from drift_presets.py
    - calls generate_X_n_list(preset) to produce X_n_list
    - calls run_one_line(X_n_list) or run_alignment_pipeline_kfold(X_n_list, ...)

This file is the only place where experiments are launched.
