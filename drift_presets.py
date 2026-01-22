from dataclasses import dataclass

ROT_SCALE      = 5e-6      # validated rotation strength
SHEAR_SCALE    = 0.06      # last stable shear before blow-up
LAG_MAX        = 1         # validated lag regime
MIX_DRIFT      = 0.7       # validated mixing-drift regime
DROPOUT_P      = 0.2  

@dataclass
class SimConfig: 
    # core sizes
    n_days: int = 5
    n_trials: int = 15
    T: int = 150
    d_latent: int = 5
    n_ch: int = 64
    seed: int = 1

    # latent drift
    rot_scale: float = 0.0
    shear_scale: float = 0.0
    lag_max: int = 0

    # observation drift
    mix_drift_scale: float = 0.0
    dropout_p: float = 0.0

    # noise
    obs_rho: float = 0.97
    obs_scale: float = 0.01
    cm_rho: float = 0.995
    cm_scale: float = 0.02


    # # noise
    # obs_rho: float = 0.0
    # obs_scale: float = 0.0
    # cm_rho: float = 0.0
    # cm_scale: float = 0.0




# -------- PRESETS --------

PURE_ROTATION = SimConfig(
    rot_scale=ROT_SCALE, 
    #.5: 
)

LAG_ONLY = SimConfig(
    seed=1,
    rot_scale=0.0,
    shear_scale=0.0,
    lag_max=LAG_MAX,
    mix_drift_scale=0.0,
    dropout_p=0.0,
)

ROT_PLUS_LAG = SimConfig(
    seed=1,
    rot_scale=ROT_SCALE, 
    shear_scale=0.0, 
    lag_max=LAG_MAX,  
    mix_drift_scale=0.0,
    dropout_p=0.0,
)

SHEAR_ONLY = SimConfig(
    rot_scale=0.000,
    shear_scale=SHEAR_SCALE, 
    lag_max=0,
    mix_drift_scale=0.0,
    dropout_p=0.0,
)

MIXING_DRIFT = SimConfig(
    rot_scale=0.000,
    shear_scale=0.000,
    lag_max=0,
    mix_drift_scale=MIX_DRIFT, 
    #0.01 → 0.03 → 0.06 → 0.1
    #The target signature is raw ≫ baseline, proc ≈ raw, lag ≈ raw, ridge_direct ≪ raw. Once you hit that regime without exploding variance or k, lock that value. 
    dropout_p=0.0, 
)
#mix_drift_scale = 0.01 → 0.03 → 0.06 → 0.1. The target signature is raw ≫ baseline, proc ≈ raw, lag ≈ raw, ridge_direct ≪ raw. 

CHANNEL_DROPOUT = SimConfig(
    rot_scale=0.000,
    shear_scale=0.000,
    lag_max=0,
    mix_drift_scale=0.0,
    dropout_p=DROPOUT_P, 
)

HARD_COMBO = SimConfig(
    rot_scale=ROT_SCALE,
    shear_scale=SHEAR_SCALE,
    lag_max=LAG_MAX,
    mix_drift_scale=MIX_DRIFT,
    dropout_p=DROPOUT_P,
)


#The process is: pick one preset, increase only its intended drift 
# knob until raw error rises; then verify the intended alignment stage 
# reduces error strongly while irrelevant stages do little; then lock 
# that knob and repeat for the next preset; only after all presets prod
# uce clean “signatures” do you train a guessing model to map observed deltas/strengths to drift type.


# rotation_strength(R) measures deviation from identity: ||R − I||.

# ridge_strength(W_res) measures how strong the non-orthogonal correction is: ||W_res||.

# Averaging across folds stabilizes these as diagnostic statistics, not training objectives.



# ---- drift knobs (how the underlying latent->channel relationship changes across days) ----

# rot_scale=0.0001,        # latent-space rotation magnitude per day; creates orthogonal drift (axes rotate)
# shear_scale=0.0001,      # latent-space shear magnitude per day; creates non-orthogonal drift (axes skew/stretch)
# lag_max=20,              # max integer time shift applied per day via np.roll; creates temporal misalignment
# mix_drift_scale=0.0,     # per-day drift of the mixing matrix W (latent->channels); 0.0 means W is fixed across days
# dropout_p=0.2,           # fraction of channels “dropped” (columns of W set to 0) per day; simulates electrode loss

# # ---- noise knobs (what gets added on top of the signal) ----

# obs_rho=0.97,            # AR(1) temporal correlation for per-channel noise; higher = smoother noise over time
# obs_scale=0.05,          # standard deviation of per-channel noise innovations; higher = noisier channels
# cm_rho=0.995,            # AR(1) temporal correlation for the common-mode noise (shared across all channels)
# cm_scale=0.08,           # standard deviation of common-mode noise; higher = bigger shared artifacts across channels
