# scripts/kona/__main__.py
import numpy as np
from scripts.kona.knob_predict import build_drift_type_dataset, train_drift_type_model

def run_once(include_diagnostics: bool, seed: int = 0, n_per_class: int = 40):
    rng = np.random.default_rng(seed)
    X, y = build_drift_type_dataset(
        rng,
        n_per_class=n_per_class,
        include_diagnostics=include_diagnostics,
        include_abs=True,
        include_deltas=True,
        include_std=False,
    )
    print(f"\n=== include_diagnostics={include_diagnostics} ===")
    print("X.shape:", X.shape, "y.shape:", y.shape)
    _scaler, _clf = train_drift_type_model(X, y, seed=seed)  # this prints mean_acc/std_acc

def main():
    run_once(include_diagnostics=False, seed=0, n_per_class=40)
    run_once(include_diagnostics=True,  seed=0, n_per_class=40)

if __name__ == "__main__":
    main()
