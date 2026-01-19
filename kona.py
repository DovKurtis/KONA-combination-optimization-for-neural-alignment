from sklearn.linear_model import LogisticRegression
import numpy as np 
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
import matplotlib as mat 
import matplotlib.pyplot as plt


# -----------------------
# PARAMETERS
# -----------------------
T = 500          # timepoints
d_latent = 5     # latent dimensionality
n_channels_d_real = 84  # observed channels
noise_std = 0.05
rng = np.random.default_rng(0)

# -----------------------
# 1. LATENT TRAJECTORY Z(t)
# shape: (500, 5)
# -----------------------
t = np.linspace(0, 2*np.pi, T)

#(500, 5) 
Z = np.stack([
    np.sin(t),
    np.cos(t),
    np.sin(2*t),
    np.cos(2*t),
    t / t.max()
], axis=1)


# 4-class labels from quadrant of (Z[:,0], Z[:,1])  -> values 0..3, shape (T,)
y = (Z[:,0] > 0).astype(int) + 2*(Z[:,1] > 0).astype(int)

# # -----------------------
# # 2. PROJECTION MATRIX (DAY 1)
# # shape: (5, 84) each of the 84 channels is a linear mixture of the d latent state components
W_1 = rng.standard_normal((d_latent, n_channels_d_real))

# # 3. OBSERVED DATA (DAY 1)
# # shape: (500, 84)
X_1 = Z @ W_1 + noise_std * rng.standard_normal((T, n_channels_d_real))
print("Day 1 data shape:", X_1.shape)

# # -----------------------
# # 2. PROJECTION MATRIX (DAY 2)
# # shape: (5, 84) each of the 84 channels is a linear mixture of the d latent state components
W_2 = rng.standard_normal((d_latent, n_channels_d_real))

# # 3. OBSERVED DATA (DAY 1)
# # shape: (500, 84)
X_2 = Z @ W_2 + noise_std * rng.standard_normal((T, n_channels_d_real))
print("Day 2 data shape:", X_2.shape)




def channel_mask_weights(X, var_pct=1, eps=1e-4, flat_frac=0.95, alpha=1.0, tiny=1e-12):
    var = X.var(axis=0)
    good = (var > np.percentile(var, var_pct)) & ((np.abs(np.diff(X, axis=0)) < eps).mean(axis=0) < flat_frac)
    std = X.std(axis=0) + tiny
    w = (1.0 / std) ** alpha          # continuous reweight
    w[~good] = 0.0                    # dropout = 0
    return w     




X_n = [X_1, X_2]


def best_lag_align(Z, Z0, max_lag=20):
    best_err = np.inf
    best_l = 0
    for l in range(-max_lag, max_lag + 1):
        if l > 0:
            A, B = Z[l:], Z0[:-l]
        elif l < 0:
            A, B = Z[:l], Z0[-l:]
        else:
            A, B = Z, Z0
        err = np.linalg.norm(A - B, ord="fro")
        if err < best_err:
            best_err, best_l = err, l
        
    if best_l > 0:
        return Z[best_l:], Z0[:-best_l]
    elif best_l < 0:
        return Z[:best_l], Z0[-best_l:]
    else:
        return Z, Z0









def dtw_align_time(Z, Z0):
    # DTW on a 1D summary to get a warping path, then warp Z to Z0’s timeline
    a = Z.mean(axis=1)      # (T,)
    b = Z0.mean(axis=1)     # (T,)
    path = dtw.warping_path(a, b)   # list of (i,j)
    # for each j in Z0, average all i that map to it
    T0 = Z0.shape[0]
    buckets = [[] for _ in range(T0)]
    for i, j in path:
        buckets[j].append(i)
    Zw = np.vstack([Z[b].mean(axis=0) if len(b)>0 else Z[min(j, Z.shape[0]-1)]
                    for j,b in enumerate(buckets)])
    return Zw, Z0






def run_pipeline(X_n, var_pct=1, eps=1e-4, flat_frac=0.95, alpha=1, k=10, lambda_=1.0):
    X_0 = X_n[0]
    mask = channel_mask_weights(X_0, var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha)
    X_0 = X_0 * mask 
    sc0 = StandardScaler().fit(X_0)
    pca0 = PCA(n_components=k, random_state=0).fit(sc0.transform(X_0))
    Z0 = PCA(n_components=k, random_state=0).fit_transform(sc0.transform(X_0))  # (500, k)
    Z0 = pca0.transform(sc0.transform(X_0))
    clf = LogisticRegression(max_iter=2000).fit(Z0, y[:len(Z0)])

    by_method = {"lag": [], "proc": [], "ridge": [], "dtw": []} 

    scores=[]
    for n in range(1,len(X_n)): 
        X = X_n[n].copy()
        mask = channel_mask_weights(X, var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha)
        X = X * mask 

        sc = StandardScaler().fit(X)
        Z = PCA(n_components=k, random_state=0).fit_transform(sc.transform(X))  # (500, k)

        Z  = pca0.transform(sc0.transform(X))  # <-- day2 projected into day1 PCA space (crucial)

        # raw mismatch (no alignment)
        err_raw = np.linalg.norm(Z - Z0, ord="fro")

        # lag only
        ZL, Z0L = best_lag_align(Z, Z0)
        err_lag = np.linalg.norm(ZL - Z0L, ord="fro")
        s_lag = err_raw / err_lag

        # DTW 
        ZLdtw, Z0Ldtw = dtw_align_time(ZL, Z0L) 
        err_dtw = np.linalg.norm(ZLdtw - Z0Ldtw, ord="fro")
        s_dtw = err_raw / err_dtw   


# 3) FIX: plot per method (lag/proc/ridge/dtw) by computing Z_aligned for each and running the same loop
# Next step: create dict {"lag":ZL,"proc":ZL_proc,"ridge":ZL_ridge,"dtw":ZLdtw} and loop over it to plot 4 curves.

        window_size = 50
        T_test = len(ZLdtw)
        accs_t = np.zeros(T_test - window_size + 1)

        T0L = len(Z0Ldtw)   # == len(ZLdtw)
        accs_t = np.zeros(T0L - window_size + 1)
        y0L = y[:len(Z0Ldtw)]
        for t in range(len(accs_t)):
            accs_t[t] = clf.score(ZLdtw[t:t+window_size], y0L[t:t+window_size])

        for t in range(len(accs_t)):
            accs_t[t] = clf.score(ZLdtw[t:t+window_size], y[t:t+window_size])

        plt.figure(figsize=(10, 4))
        plt.plot(accs_t)
        plt.xlabel('Timepoint')
        plt.ylabel('Decoding Accuracy')
        plt.title(f'Sliding Window Accuracy (window={window_size})')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.show()

        # proc after lag (still compare to err_raw)
        R, _ = orthogonal_procrustes(ZL, Z0L)
        ZL_proc = ZL @ R
        err_lag_proc = np.linalg.norm(ZL_proc - Z0L, ord="fro")
        s_proc = err_raw / err_lag_proc

        # ridge after lag (still compare to err_raw)
        ridge = Ridge(alpha=lambda_, fit_intercept=False).fit(ZL, Z0L)
        ZL_ridge = ridge.predict(ZL)
        err_lag_ridge = np.linalg.norm(ZL_ridge - Z0L, ord="fro")
        s_ridge = err_raw / err_lag_ridge


        # aggregate per-method across (days x folds if you add more days)
        by_method["lag"].append(s_lag)
        by_method["proc"].append(s_proc)
        by_method["ridge"].append(s_ridge)
        by_method["dtw"].append(s_dtw) 
    
    means = {m: float(np.mean(v)) for m, v in by_method.items()}
    return means 

#ie {'lag': 1.42, 'proc': 2.87, 'ridge': 3.91}# Yes. Do ONE grid loop and update best for all 3 methods at once.
import itertools


import itertools
import numpy as np

def best_params_all_methods_onepass(
    X_n, var_pcts=1, epss=1e-4, flat_fracs=0.95, alphas=1.0, ks=10, lambdas_=1.0):
    
    var_pcts, epss, flat_fracs, alphas, ks, lambdas_ = map(np.atleast_1d, [var_pcts, epss, flat_fracs, alphas, ks, lambdas_])

    best = {
        "lag":   {"score": -np.inf, "params": None},
        "proc":  {"score": -np.inf, "params": None},
        "ridge": {"score": -np.inf, "params": None},
        "dtw":   {"score": -np.inf, "params": None},
    }

    for var_pct, eps, flat_frac, alpha, k, lambda_ in itertools.product(var_pcts, epss, flat_fracs, alphas, ks, lambdas_):
        means = run_pipeline(
            X_n,
            var_pct=float(var_pct),
            eps=float(eps),
            flat_frac=float(flat_frac),
            alpha=float(alpha),
            k=int(k),
            lambda_=float(lambda_),
        )

        if means["lag"] > best["lag"]["score"]:
            best["lag"] = {"score": means["lag"],
                           "params": dict(var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha, k=k)}
            
        if means["proc"] > best["proc"]["score"]:
            best["proc"] = {"score": means["proc"],
                            "params": dict(var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha, k=k)}
            
        if means["ridge"] > best["ridge"]["score"]:
            best["ridge"] = {"score": means["ridge"],
                             "params": dict(var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha, k=k, lambda_=lambda_)}
            
        if means["dtw"] > best["dtw"]["score"]:
            best["dtw"] = {"score": means["dtw"],
                           "params": dict(var_pct=var_pct, eps=eps, flat_frac=flat_frac, alpha=alpha, k=k)}


    return best

best = best_params_all_methods_onepass(
    X_n, 
    var_pcts=[1],
    epss=[1e-4],
    flat_fracs=[0.95],
    alphas=[1.0],
    ks=[5, 10],
    lambdas_=[1.0]
)
print(best)
