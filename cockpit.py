from kona.sim_core import generate_X_n_list
from scripts.kona.drift_presets import HARD_COMBO #<——any preset
from scripts.kona.align_X_n_list_and_evaluate import run_one_line, run_alignment_pipeline_kfold

X_n_list = generate_X_n_list(HARD_COMBO)

# sweep then run full pipeline (recommended)
best = run_one_line(X_n_list)
print("FINAL BEST PARAMS:", best)

# run only the full pipeline without sweep
# run_alignment_pipeline_kfold(
#     X_n_list,
#     k=7,
#     n_splits=5,
#     seed=0,
#     alpha=1.0,
#     lambda_res=300.0,
#     max_lag=25,
# )
