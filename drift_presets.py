from dataclasses import dataclass

@dataclass
class SimConfig:
    # core sizes
    n_days: int = 5
    n_trials: int = 100
    T: int = 500
    d_latent: int = 5
    n_ch: int = 84
    seed: int = 0

    # latent drift
    rot_scale: float = 0.0
    shear_scale: float = 0.0
    lag_max: int = 0

    # observation drift
    mix_drift_scale: float = 0.0
    dropout_p: float = 0.0

    # noise
    obs_rho: float = 0.97
    obs_scale: float = 0.05
    cm_rho: float = 0.995
    cm_scale: float = 0.08


# -------- PRESETS --------

PURE_ROTATION = SimConfig(
    rot_scale=0.35,
)

ROTATION_PLUS_LAG = SimConfig(
    rot_scale=0.35,
    lag_max=15,
)

SHEAR_ONLY = SimConfig(
    shear_scale=0.25,
)

MIXING_DRIFT = SimConfig(
    mix_drift_scale=0.05,
)

CHANNEL_DROPOUT = SimConfig(
    dropout_p=0.25,
)

HARD_COMBO = SimConfig(
    rot_scale=0.3,
    shear_scale=0.2,
    lag_max=20,
    mix_drift_scale=0.05,
    dropout_p=0.2,
)
