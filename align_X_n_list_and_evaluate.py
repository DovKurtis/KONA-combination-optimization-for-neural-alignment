
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# -------------

def flatten_day(X_day):
    n_trials, T, n_ch = X_day.shape
    return X_day.reshape(n_trials * T, n_ch)

def channel_mask_weights_trials(X_day, var_pct=1, eps=1e-4, flat_frac=0.95, alpha=1.0, tiny=1e-12):
    n_trials, T, n_ch = X_day.shape
    X2 = X_day.reshape(n_trials*T, n_ch)
    var = X2.var(axis=0)
    std = X2.std(axis=0) + tiny
    flat = (np.abs(np.diff(X_day, axis=1)) < eps).mean(axis=(0,1))
    good = (var > np.percentile(var, var_pct)) & (flat < flat_frac)
    w = (1.0 / std) ** alpha
    w[~good] = 0.0
    return w

def crop_pair(A, B, lag):
    if lag > 0:
        return A[lag:], B[:-lag]
    elif lag < 0:
        return A[:lag], B[-lag:]
    else:
        return A, B

def frob_error(A_hat, A_ref):
    return float(np.linalg.norm(A_hat - A_ref, "fro") / (np.linalg.norm(A_ref, "fro") + 1e-12))
def fit_ridge_on_trials(Zd_tr_trials, Z0_tr_trials, lambda_=1.0):
    # Zd_tr_trials, Z0_tr_trials: (n_train, T, k)
    A2 = Zd_tr_trials.reshape(-1, Zd_tr_trials.shape[-1])  # (n_train*T, k)
    B2 = Z0_tr_trials.reshape(-1, Z0_tr_trials.shape[-1])  # (n_train*T, k)
    ridge = Ridge(alpha=float(lambda_), fit_intercept=False).fit(A2, B2)
    return ridge.coef_.T  # (k, k)

def best_lag_align(A, B, max_lag=25):
    # A, B: (T, k)
    best_err = np.inf
    best_l = 0
    for l in range(-max_lag, max_lag + 1):
        if l > 0:
            A1, B1 = A[l:], B[:-l]
        elif l < 0:
            A1, B1 = A[:l], B[-l:]
        else:
            A1, B1 = A, B
        err = np.linalg.norm(A1 - B1, ord="fro")
        if err < best_err:
            best_err, best_l = err, l
    return best_l

def run_alignment_pipeline_kfold(X_n_list,k=7,n_splits=5,seed=0,var_pct=1,eps=1e-4,flat_frac=0.95,alpha=1.0,lambda_res=300.0,max_lag=25,):
    n_days, n_trials, T, _ = X_n_list.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # day0-only preprocessing pipeline
    X0_flat = flatten_day(X_n_list[0]) 
    w0 = channel_mask_weights_trials(X_n_list[0], var_pct, eps, flat_frac, alpha)
    X0m = X0_flat * w0
    sc0 = StandardScaler().fit(X0m)
    pca0 = PCA(n_components=k, random_state=0).fit(sc0.transform(X0m))

    def day_latents(day):
        Xd_flat = flatten_day(X_n_list[day])
        Xdm = Xd_flat * w0
        Zd_flat = pca0.transform(sc0.transform(Xdm))
        return Zd_flat.reshape(n_trials, T, k)

    Z_days = [day_latents(d) for d in range(n_days)]
    methods = ["raw", "lag", "proc", "lag+proc", "ridge_direct"]
    ridge_train_errs = []
    ridge_gain = []
    ridge_norms=[] 
    for day in range(1, n_days):
        scores = {m: [] for m in methods}
        for train_idx, test_idx in kf.split(np.arange(n_trials)):
            Z0_tr = Z_days[0][train_idx].mean(axis=0)   # (T,k)
            Zd_tr = Z_days[day][train_idx].mean(axis=0)
            Z0_te = Z_days[0][test_idx].mean(axis=0)
            Zd_te = Z_days[day][test_idx].mean(axis=0)

            # raw
            scores["raw"].append(frob_error(Zd_te, Z0_te))

            # proc only (no lag)
            R0, _ = orthogonal_procrustes(Zd_tr, Z0_tr)
            scores["proc"].append(frob_error(Zd_te @ R0, Z0_te))

            # lag (chosen on TRAIN means)
            lag = best_lag_align(Zd_tr, Z0_tr, max_lag=max_lag)
            Zd_te_c, Z0_te_c = crop_pair(Zd_te, Z0_te, lag)
            scores["lag"].append(frob_error(Zd_te_c, Z0_te_c))

            # lag+proc (choose lag on TRAIN, then proc on TRAIN after cropping)
            Zd_tr_c, Z0_tr_c = crop_pair(Zd_tr, Z0_tr, lag)
            R_lp, _ = orthogonal_procrustes(Zd_tr_c, Z0_tr_c)

            Zd_te_c, Z0_te_c = crop_pair(Zd_te, Z0_te, lag)
            scores["lag+proc"].append(frob_error(Zd_te_c @ R_lp, Z0_te_c))

            # ridge_direct baseline: fit on TRAIN trials, score on TEST means (no lag/proc)
            W_dir = fit_ridge_on_trials(Z_days[day][train_idx], Z_days[0][train_idx], lambda_=lambda_res)
            scores["ridge_direct"].append(frob_error(Zd_te @ W_dir, Z0_te))

                        # ridge_direct: fit on TRAIN trials

            # TEST mean score (already have)

            # NEW: TRAIN mean score (diagnostic)
            Zd_tr_mean = Z_days[day][train_idx].mean(axis=0)  # (T,k)
            Z0_tr_mean = Z_days[0][train_idx].mean(axis=0)
            ridge_train_errs.append(frob_error(Zd_tr_mean @ W_dir, Z0_tr_mean))

            # NEW: ridge-vs-raw improvement on TEST (diagnostic)
            raw_err = scores["raw"][-1]
            ridge_err = scores["ridge_direct"][-1]
            ridge_gain.append(float(raw_err - ridge_err))

            # NEW: keep norm (you already do)
            ridge_norms.append(float(np.linalg.norm(W_dir, ord="fro")))

            


        # print “table row” for this day
        parts = [f"day {day}"]
        for m in methods:
            parts.append(f"{m} {np.mean(scores[m]):.4f}±{np.std(scores[m]):.4f}")
    
              
def fit_procrustes_on_trials(Zd_tr_trials, Z0_tr_trials):
    # Zd_tr_trials, Z0_tr_trials: (n_train, T, k)
    # fit on TRAIN MEAN trajectory (same as your proc baseline)
    Zd_tr_mean = Zd_tr_trials.mean(axis=0)  # (T,k)
    Z0_tr_mean = Z0_tr_trials.mean(axis=0)  # (T,k)
    R, _ = orthogonal_procrustes(Zd_tr_mean, Z0_tr_mean)
    return R  # (k,k)

def sweep_k_lag_fast(
    X_n_list,
    k_list=(5, 7, 8, 10, 12),
    n_splits=3,
    seed=0,
    var_pct=1,
    eps=1e-4,
    flat_frac=0.95,
    alpha=1.0,
    max_lag=25, 
):
    n_days, n_trials, T, _ = X_n_list.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # GLOBAL weights from day0
    X0_flat = flatten_day(X_n_list[0])
    w0 = channel_mask_weights_trials(X_n_list[0], var_pct, eps, flat_frac, alpha)
    X0m = X0_flat * w0

    best = {"lag_mean": np.inf, "k": None}

    results = []

    for k in k_list:
        sc0 = StandardScaler().fit(X0m)
        pca0 = PCA(n_components=k, random_state=0).fit(sc0.transform(X0m))

        def day_latents(day):
            Xd_flat = flatten_day(X_n_list[day])
            Xdm = Xd_flat * w0
            Zd_flat = pca0.transform(sc0.transform(Xdm))
            return Zd_flat.reshape(n_trials, T, k)

        Z_days = [day_latents(d) for d in range(n_days)]

        lag_scores = []
        for day in range(1, n_days):
            for train_idx, test_idx in kf.split(np.arange(n_trials)):
                # fit procrustes on TRAIN means

                # score on TEST means
                Zd_te_mean = Z_days[day][test_idx].mean(axis=0)
                Z0_te_mean = Z_days[0][test_idx].mean(axis=0)
                
                lag = best_lag_align(Zd_te_mean, Z0_te_mean, max_lag=max_lag)
                Zd_c, Z0_c = crop_pair(Zd_te_mean, Z0_te_mean, lag)
                lag_scores.append(frob_error(Zd_c, Z0_c))


        lag_mean = float(np.mean(lag_scores))
        results.append({"k": int(k), "lag_mean": lag_mean})

        if lag_mean < best["lag_mean"]:
            best = {"lag_mean": lag_mean, "k": int(k)}

    print("\nBEST (PROC):")
    print(best)
    return best, results, best["k"] 

def run_proc_tuning_report(
    X_n_list,
    k_list=(5, 7, 8, 10, 12),
    n_splits_sweep=3,
    n_splits_eval=5,
    seed=0,
    var_pct=1,
    eps=1e-4,
    flat_frac=0.95,
    alpha=1.0,
):
    # 1) sweep k for best 
    best, _, best_k = sweep_k_lag_fast(
        X_n_list,
        k_list=k_list,
        n_splits=n_splits_sweep,
        seed=seed,
        var_pct=var_pct,
        eps=eps,
        flat_frac=flat_frac,
        alpha=alpha,
    )

    # 2) run per-day proc table (prints for ALL days in X_n_list)
    run_alignment_pipeline_kfold(
        X_n_list,
        k=best["k"],
        n_splits=n_splits_eval,
        seed=seed,
        var_pct=var_pct,
        eps=eps,
        flat_frac=flat_frac,
        alpha=alpha,
        lambda_res=0.0,   # ignored by proc, but required by signature
    )
    return best, best_k

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# ... keep your existing helpers: flatten_day, channel_mask_weights_trials, crop_pair, frob_error,
# fit_ridge_on_trials, best_lag_align, etc.

def eval_alignment_scores(
    X_n_list,
    k=7,
    n_splits=5,
    seed=0,
    var_pct=1,
    eps=1e-4,
    flat_frac=0.95,
    alpha=1.0,
    lambda_res=300.0,
    max_lag=25,
):
    n_days, n_trials, T, _ = X_n_list.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X0_flat = flatten_day(X_n_list[0])
    w0 = channel_mask_weights_trials(X_n_list[0], var_pct, eps, flat_frac, alpha)
    X0m = X0_flat * w0
    sc0 = StandardScaler().fit(X0m)
    pca0 = PCA(n_components=k, random_state=0).fit(sc0.transform(X0m))

    def day_latents(day):
        Xd_flat = flatten_day(X_n_list[day])
        Xdm = Xd_flat * w0
        Zd_flat = pca0.transform(sc0.transform(Xdm))
        return Zd_flat.reshape(n_trials, T, k)

    Z_days = [day_latents(d) for d in range(n_days)]

    methods = ["raw", "lag", "proc", "lag+proc", "ridge_direct"]
    per_day = {}
    extras = {}  # NEW diagnostics

    for day in range(1, n_days):
        scores = {m: [] for m in methods}

        # NEW diagnostics collectors
        lags_this_day = []
        ridge_norms = []
        proc_residuals = []

        ridge_train_errs = []
        ridge_gain = []


        for train_idx, test_idx in kf.split(np.arange(n_trials)):
            Z0_tr = Z_days[0][train_idx].mean(axis=0)   # (T,k)
            Zd_tr = Z_days[day][train_idx].mean(axis=0) # (T,k)
            Z0_te = Z_days[0][test_idx].mean(axis=0)    # (T,k)
            Zd_te = Z_days[day][test_idx].mean(axis=0)  # (T,k)

            # raw
            scores["raw"].append(frob_error(Zd_te, Z0_te))

            # proc (no lag)
            R0, _ = orthogonal_procrustes(Zd_tr, Z0_tr)
            scores["proc"].append(frob_error(Zd_te @ R0, Z0_te))

            # NEW: proc fit residual on TRAIN means (how well proc fits day drift)
            proc_residuals.append(frob_error(Zd_tr @ R0, Z0_tr))

            # lag (chosen on TRAIN means)
            lag = best_lag_align(Zd_tr, Z0_tr, max_lag=max_lag)
            lags_this_day.append(abs(int(lag)))  # NEW

            Zd_te_c, Z0_te_c = crop_pair(Zd_te, Z0_te, lag)
            scores["lag"].append(frob_error(Zd_te_c, Z0_te_c))

            # lag+proc
            Zd_tr_c, Z0_tr_c = crop_pair(Zd_tr, Z0_tr, lag)
            R_lp, _ = orthogonal_procrustes(Zd_tr_c, Z0_tr_c)
            Zd_te_c, Z0_te_c = crop_pair(Zd_te, Z0_te, lag)
            scores["lag+proc"].append(frob_error(Zd_te_c @ R_lp, Z0_te_c))

            # ridge_direct
            W_dir = fit_ridge_on_trials(Z_days[day][train_idx], Z_days[0][train_idx], lambda_=lambda_res)
            scores["ridge_direct"].append(frob_error(Zd_te @ W_dir, Z0_te))

            # NEW diagnostics
            Zd_tr_mean = Z_days[day][train_idx].mean(axis=0)  # (T,k)
            Z0_tr_mean = Z_days[0][train_idx].mean(axis=0)
            ridge_train_errs.append(frob_error(Zd_tr_mean @ W_dir, Z0_tr_mean))

            raw_err   = scores["raw"][-1]
            ridge_err = scores["ridge_direct"][-1]
            ridge_gain.append(float(raw_err - ridge_err))

            ridge_norms.append(float(np.linalg.norm(W_dir, ord="fro")))  # if not already there

        per_day[day] = {m: (float(np.mean(scores[m])), float(np.std(scores[m]))) for m in methods}

        extras[day] = {
            "mean_lag": float(np.mean(lags_this_day)) if lags_this_day else 0.0,
            "ridge_norm": float(np.mean(ridge_norms)) if ridge_norms else 0.0,
            "proc_residual": float(np.mean(proc_residuals)) if proc_residuals else 0.0,
        }

        # ADD THIS EXACTLY HERE (end of day loop, after extras[day] exists)
        extras[day].update({
            "ridge_train_err": float(np.mean(ridge_train_errs)) if ridge_train_errs else 0.0,
            "ridge_gain": float(np.mean(ridge_gain)) if ridge_gain else 0.0,
        })


           
    return {"methods": methods, "per_day": per_day, "extras": extras}


