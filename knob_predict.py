# scripts/kona/knob_predict.py

from dataclasses import asdict
from itertools import product
import numpy as np

from .drift_presets import SimConfig, HARD_COMBO
from .sim_core import generate_X_n_list
from .align_X_n_list_and_evaluate import eval_alignment_scores

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

METHODS = ["raw", "lag", "proc", "lag+proc", "ridge_direct"]
KNOB_KEYS = ["rot_scale","shear_scale","lag_max","mix_drift_scale","dropout_p"]

# ---------- ONE shared feature extractor ----------
def feature_vector(
    score_blob,
    use_methods=("lag","proc","lag+proc","ridge_direct"),
    include_abs=True,
    include_deltas=True,
    include_std=False,
    include_diagnostics=False,
):
    per_day = score_blob["per_day"]
    extras  = score_blob.get("extras", {})
    days = sorted(per_day.keys())

    rows = []
    for day in days:
        mdict = per_day[day]
        raw_mean = float(mdict["raw"][0])
        parts = []

        if include_abs:
            parts.extend([float(mdict[m][0]) for m in METHODS])  # 5
        if include_deltas:
            parts.extend([raw_mean - float(mdict[m][0]) for m in use_methods])  # 4
        if include_std:
            parts.extend([float(mdict[m][1]) for m in METHODS])  # 5
        if include_diagnostics:
            ex = extras.get(day, {})
            parts.append(float(ex.get("mean_lag", 0.0)))
            parts.append(float(ex.get("ridge_norm", 0.0)))
            parts.append(float(ex.get("proc_residual", 0.0)))
            parts.append(float(ex.get("ridge_train_err", 0.0)))
            parts.append(float(ex.get("ridge_gain", 0.0)))

        rows.append(parts)

    return np.asarray(rows, float).reshape(-1)  # (D,)


# ---------- CLASSIFICATION (recommended to keep) ----------
DRIFT_CLASS_NAMES = ["rotation", "shear", "lag", "mix", "dropout"]
C_ROT, C_SHEAR, C_LAG, C_MIX, C_DROP = range(5)

def build_drift_type_dataset(
    rng,
    n_per_class=50,
    base_cfg=HARD_COMBO,
    rot_range=(1e-6, 5e-5),
    shear_range=(0.02, 0.10),
    lag_range=(3, 20),
    mix_range=(0.02, 0.15),
    drop_range=(0.05, 0.40),
    include_abs=True, include_deltas=True, include_std=False, include_diagnostics=False,
):
    X_rows, y_rows = [], []

    def clean_cfg():
        cfg = SimConfig(**asdict(base_cfg))
        cfg.seed = int(rng.integers(0, 2**31 - 1))
        cfg.rot_scale = 0.0
        cfg.shear_scale = 0.0
        cfg.lag_max = 0
        cfg.mix_drift_scale = 0.0
        cfg.dropout_p = 0.0
        return cfg

    for _ in range(n_per_class):
        cfg = clean_cfg(); cfg.rot_scale = float(rng.uniform(*rot_range))
        Xn = generate_X_n_list(cfg); blob = eval_alignment_scores(Xn)
        X_rows.append(feature_vector(
            blob,
            include_abs=include_abs,
            include_deltas=include_deltas,
            include_std=include_std,
            include_diagnostics=include_diagnostics,
        ))
        y_rows.append(C_ROT)         
        del Xn, blob

        cfg = clean_cfg(); cfg.shear_scale = float(rng.uniform(*shear_range))
        Xn = generate_X_n_list(cfg); blob = eval_alignment_scores(Xn)
        X_rows.append(feature_vector(
            blob,
            include_abs=include_abs,
            include_deltas=include_deltas,
            include_std=include_std,
            include_diagnostics=include_diagnostics,
        ))        
        y_rows.append(C_SHEAR)  
        del Xn, blob

        cfg = clean_cfg(); cfg.lag_max = int(rng.integers(lag_range[0], lag_range[1] + 1))
        Xn = generate_X_n_list(cfg); blob = eval_alignment_scores(Xn)
        X_rows.append(feature_vector(
            blob,
            include_abs=include_abs,
            include_deltas=include_deltas,
            include_std=include_std,
            include_diagnostics=include_diagnostics,
        ))        
        y_rows.append(C_LAG)
        del Xn, blob

        cfg = clean_cfg(); cfg.mix_drift_scale = float(rng.uniform(*mix_range))
        Xn = generate_X_n_list(cfg); blob = eval_alignment_scores(Xn)
        X_rows.append(feature_vector(
            blob,
            include_abs=include_abs,
            include_deltas=include_deltas,
            include_std=include_std,
            include_diagnostics=include_diagnostics,
        ))        
        y_rows.append(C_MIX)
        del Xn, blob

        cfg = clean_cfg(); cfg.dropout_p = float(rng.uniform(*drop_range))
        Xn = generate_X_n_list(cfg); blob = eval_alignment_scores(Xn)
        X_rows.append(feature_vector(
            blob,
            include_abs=include_abs,
            include_deltas=include_deltas,
            include_std=include_std,
            include_diagnostics=include_diagnostics,
        ))        
        y_rows.append(C_DROP)  
        del Xn, blob

    return np.asarray(X_rows, float), np.asarray(y_rows, int)  # (N,D), (N,)

def train_drift_type_model(X, y, seed=0):
    X = np.asarray(X, float)
    y = np.asarray(y, int)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(scaler.transform(X[tr]), y[tr])

        pred = clf.predict(scaler.transform(X[te]))
        acc = accuracy_score(y[te], pred)
        accs.append(acc)

        print("fold", fold, "acc:", acc)
        print("confusion:\n", confusion_matrix(y[te], pred))

    print("mean_acc:", float(np.mean(accs)), "std_acc:", float(np.std(accs)))

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs")
    clf.fit(scaler.transform(X), y)
    return scaler, clf



def predict_drift_type_from_raw(scaler, clf, X_n_list,
                                include_abs=True, include_deltas=True, include_std=False, include_diagnostics=False):
    blob = eval_alignment_scores(X_n_list)
    x = feature_vector(blob, include_abs, include_deltas, include_std, include_diagnostics)[None, :]
    proba = clf.predict_proba(scaler.transform(x))[0]   # (5,)
    lab = int(np.argmax(proba))
    return lab, DRIFT_CLASS_NAMES[lab], proba
