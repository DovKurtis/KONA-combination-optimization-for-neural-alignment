
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
def run_alignment_pipeline_kfold(
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

    # GLOBAL day0 weights + scaler + PCA
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

    for day in range(1, n_days):
        raw_l, final_l, lagvals = [], [], []

        for train_idx, test_idx in kf.split(np.arange(n_trials)):
            Z0_tr = Z_days[0][train_idx].mean(axis=0)
            Zd_tr = Z_days[day][train_idx].mean(axis=0)
            Z0_te = Z_days[0][test_idx].mean(axis=0)
            Zd_te = Z_days[day][test_idx].mean(axis=0)

            raw_l.append(frob_error(Zd_te, Z0_te))

            # 1) lag from TRAIN means
            lag = best_lag_align(Zd_tr, Z0_tr, max_lag=max_lag)
            lagvals.append(lag)

            def crop_pair(A, B, lag):
                if lag > 0:
                    return A[lag:], B[:-lag]
                elif lag < 0:
                    return A[:lag], B[-lag:]
                else:
                    return A, B

            ZdL_tr, Z0L_tr = crop_pair(Zd_tr, Z0_tr, lag)
            ZdL_te, Z0L_te = crop_pair(Zd_te, Z0_te, lag)

            # 2) procrustes rotation from TRAIN means
            R, _ = orthogonal_procrustes(ZdL_tr, Z0L_tr)

            # apply rotation
            ZdR_tr = ZdL_tr @ R
            ZdR_te = ZdL_te @ R

            # 3) ridge on RESIDUALS using TRAIN TRIALS
            Zd_trials = Z_days[day][train_idx]  # (n_train,T,k)
            Z0_trials = Z_days[0][train_idx]

            # crop trials to match lag
            if lag > 0:
                A = (Zd_trials[:, lag:, :] @ R)         # rotated
                B = Z0_trials[:, :-lag, :]
            elif lag < 0:
                A = (Zd_trials[:, :lag, :] @ R)
                B = Z0_trials[:, -lag:, :]
            else:
                A = (Zd_trials @ R)
                B = Z0_trials

            # residual targets in latent space
            # want: (A + A@W_res) ~= B  => learn W_res to map A -> (B - A)
            A2 = A.reshape(-1, k)
            Y2 = (B.reshape(-1, k) - A2)

            W_res = Ridge(alpha=float(lambda_res), fit_intercept=False).fit(A2, Y2).coef_.T  # (k,k)

            # apply residual correction on TEST
            Zd_final_te = ZdR_te + (ZdR_te @ W_res)

            final_l.append(frob_error(Zd_final_te, Z0L_te))

        print(
            f"day {day} | "
            f"raw {np.mean(raw_l):.4f}±{np.std(raw_l):.4f} | "
            f"FINAL {np.mean(final_l):.4f}±{np.std(final_l):.4f} | "
            f"lag_mean={np.mean(lagvals):.2f}"
        )


def sweep_k_lambda_fast(
    X_n_list,
    k_list=(5, 8, 10, 12, 15),
    lambda_list=(1e-6, 1e-4, 1e-2, 1.0, 100.0),
    n_splits=3,          # keep small for speed; change to 5 after you pick best
    seed=0,
    var_pct=1,
    eps=1e-4,
    flat_frac=0.95,
    alpha=1.0,
):
    n_days, n_trials, T, _ = X_n_list.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # GLOBAL weights from day0 (your "no mismatch" rule)
    X0_flat = flatten_day(X_n_list[0])
    w0 = channel_mask_weights_trials(X_n_list[0], var_pct, eps, flat_frac, alpha)
    X0m = X0_flat * w0

    best = {"ridge_mean": np.inf, "k": None, "lambda_": None}
    results = []

    for k in k_list:
        # fit scaler + PCA once per k (day0)
        sc0 = StandardScaler().fit(X0m)
        pca0 = PCA(n_components=k, random_state=0).fit(sc0.transform(X0m))

        def day_latents(day):
            Xd_flat = flatten_day(X_n_list[day])
            Xdm = Xd_flat * w0
            Zd_flat = pca0.transform(sc0.transform(Xdm))
            return Zd_flat.reshape(n_trials, T, k)

        Z_days = [day_latents(d) for d in range(n_days)]

        for lambda_ in lambda_list:
            ridge_scores = []

            for day in range(1, n_days):
                for train_idx, test_idx in kf.split(np.arange(n_trials)):
                    # train mean (for proc)
                    Z0_tr_mean = Z_days[0][train_idx].mean(axis=0)
                    Zd_tr_mean = Z_days[day][train_idx].mean(axis=0)

                    # test mean (for scoring)
                    Z0_te_mean = Z_days[0][test_idx].mean(axis=0)
                    Zd_te_mean = Z_days[day][test_idx].mean(axis=0)

                    # ridge fit on train trials
                    W_ridge = fit_ridge_on_trials(
                        Z_days[day][train_idx],
                        Z_days[0][train_idx],
                        lambda_=lambda_,
                    )

                    ridge_scores.append(frob_error(Zd_te_mean @ W_ridge, Z0_te_mean))

            ridge_mean = float(np.mean(ridge_scores))
            results.append({"k": k, "lambda_": float(lambda_), "ridge_mean": ridge_mean})

            if ridge_mean < best["ridge_mean"]:
                best = {"ridge_mean": ridge_mean, "k": k, "lambda_": float(lambda_)}

            print(f"k={k} lambda_={lambda_:g} ridge_mean={ridge_mean:.4f} | best k={best['k']} lambda_={best['lambda_']:g} ridge={best['ridge_mean']:.4f}")

    print("\nBEST:")
    print(best)
    return best, results

def run_one_line(X_n_list):
    best, _ = sweep_k_lambda_fast(
        X_n_list,
        k_list=(5, 7, 8, 10, 12),
        lambda_list=(1e-4, 1e-2, 1.0, 30.0, 100.0, 175.0, 300.0),
        n_splits=3,
        seed=0,
        alpha=1.0,
    )

    run_alignment_pipeline_kfold(
    X_n_list,
    k=7,
    n_splits=5,
    seed=0,
    alpha=1.0,
    lambda_res=300.0,
    max_lag=25,
)

    return best

#proc_err should drop clearly below raw_err for every day (not occasionally), 
# and ridge_err should be substantially below proc_err; as a concrete target, 
# # aim ridge_err < 0.6 first, then < 0.4 once you tune noise/drift magnitudes
# BEST:
# {'ridge_mean': 0.6435849531190421, 'k': 7, 'lambda_': 300.0}
# day 1 | raw 1.4738±0.1048 | proc 1.1478±0.1055 | ridge 0.7291±0.0183



# {'ridge_mean': 0.6435849531190421, 'k': 7, 'lambda_': 300.0}
# day 1 | raw 1.4738±0.1048 | lag 1.3832±0.1084 | proc 1.2939±0.0721 | ridge 0.7597±0.0096 | lag_mean=12.20
# day 2 | raw 1.2182±0.0517 | lag 1.1865±0.0642 | proc 1.1360±0.1139 | ridge 0.7171±0.0170 | lag_mean=-10.80
