# Contains ONLY generation utilities (latents, drifts, observation).
# generates latent trial trajectories, applies 
# day-specific latent drifts (rotation, shear, lag),
#  mixes latents into channels, injects observation noise, 
# and returns the final (days, trials, time, channels) tensor 
#-------------GENERATES X_n_list -------------
import sys
print("SIM_CORE LOADED FROM:", __file__, "argv0:", sys.argv[0])
from scipy.linalg import expm
import numpy as np

def smooth_ar1(T, rng, rho=0.98, scale=1.0):
    
    x = np.zeros(T)
    e = rng.standard_normal(T) * scale
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x

def make_latent_trials(n_trials, T, d_latent, rng):
    Z_trials = np.zeros((n_trials, T, d_latent))
    for tr in range(n_trials):
        onset_hi = max(1, min(180, T-1))
        onset = int(rng.integers(0, onset_hi))  # or keep a min like 40 if T allows
        dur = int(rng.integers(1, min(320, T - onset) + 1))  # ensures end-onset >= 1
        end = onset + dur
        s = np.linspace(0, 1, end - onset, endpoint=False)
        base = np.stack([
            np.sin(2*np.pi*(3+2*rng.random())*s + 2*np.pi*rng.random()),
            np.cos(2*np.pi*(2+2*rng.random())*s + 2*np.pi*rng.random()),
            smooth_ar1(end-onset, rng, rho=0.97, scale=0.15),
            smooth_ar1(end-onset, rng, rho=0.99, scale=0.08),
            s,
        ], axis=1)

        amp = rng.normal(1.0, 0.25, size=(1, d_latent))
        Z_trials[tr, onset:end, :] = amp * base
        Z_trials[tr] += 0.05 * rng.standard_normal((T, d_latent))
    return Z_trials

def colored_obs_noise(T, n_ch, rng, rho=0.97, scale=0.05, cm_rho=0.995, cm_scale=0.08):
    e = rng.standard_normal((T, n_ch)) * scale
    x = np.zeros((T, n_ch))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    cm = smooth_ar1(T, rng, rho=cm_rho, scale=cm_scale)[:, None]
    return x + cm

def random_rotation_scaled(d, rng, scale):
    A = rng.standard_normal((d, d))
    S = A - A.T                      # skew-symmetric
    Q = expm(scale * S)              # true rotation, angle ∝ scale
    return Q

def random_shear_scaled(d, rng, scale):
    # near-identity non-orthogonal shear
    M = np.eye(d) + scale * rng.standard_normal((d, d))
    return M

def make_day_transforms(cfg, rng):
    # latent transforms
    R_list = [np.eye(cfg.d_latent)]
    S_list = [np.eye(cfg.d_latent)]
    for _ in range(cfg.n_days - 1):
        R_list.append(random_rotation_scaled(cfg.d_latent, rng, cfg.rot_scale) if cfg.rot_scale > 0 else np.eye(cfg.d_latent))
        S_list.append(random_shear_scaled(cfg.d_latent, rng, cfg.shear_scale) if cfg.shear_scale > 0 else np.eye(cfg.d_latent))
    if cfg.lag_max > 0:
        lag_list = [0]
        for _ in range(cfg.n_days - 1):
            l = 0
            while l == 0:
                l = int(rng.integers(-cfg.lag_max, cfg.lag_max + 1))
            lag_list.append(l)
    else:
        lag_list = [0] * cfg.n_days

    return R_list, S_list, lag_list

def make_W_per_day(cfg, rng):
    # base mixing
    W0 = rng.standard_normal((cfg.d_latent, cfg.n_ch))
    W_list = [W0]
    for _ in range(cfg.n_days - 1):
        if cfg.mix_drift_scale > 0:
            W_list.append(W_list[-1] + cfg.mix_drift_scale * rng.standard_normal((cfg.d_latent, cfg.n_ch)))
        else:
            W_list.append(W0)
    return W_list

def apply_dropout(W, dropout_p, rng):
    if dropout_p <= 0:
        return W
    W2 = W.copy()
    n_ch = W2.shape[1]
    n_drop = int(np.round(dropout_p * n_ch))
    if n_drop <= 0:
        return W2
    drop_idx = rng.choice(n_ch, size=n_drop, replace=False)
    W2[:, drop_idx] = 0.0
    return W2


def generate_X_n_list(cfg):
    rng = np.random.default_rng(cfg.seed)
    Z_trials = make_latent_trials(cfg.n_trials, cfg.T, cfg.d_latent, rng)

    R_list, S_list, lag_list = make_day_transforms(cfg, rng)
    R_list, S_list, lag_list = make_day_transforms(cfg, rng)
    W_list = make_W_per_day(cfg, rng)

    X_days = []
    for day in range(cfg.n_days):
        Zt = Z_trials.copy()

        lag = lag_list[day]
        if lag != 0:
            Zt = np.roll(Zt, shift=lag, axis=1)

        # latent transform per day
        Z_drift = Zt @ (S_list[day] @ R_list[day])  # shear then rotate

        # mixing per day (+ dropout)
        Wd = apply_dropout(W_list[day], cfg.dropout_p, rng)

        X_day = np.empty((cfg.n_trials, cfg.T, cfg.n_ch))
        for tr in range(cfg.n_trials):
            X_tr = Z_drift[tr] @ Wd
            X_tr += colored_obs_noise(cfg.T, cfg.n_ch, rng, rho=cfg.obs_rho, scale=cfg.obs_scale, cm_rho=cfg.cm_rho, cm_scale=cfg.cm_scale)
            X_day[tr] = X_tr

        X_days.append(X_day)

    return np.array(X_days, dtype=np.float32)  # (n_days, n_trials, T, n_ch)
