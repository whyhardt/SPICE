# Grid searches

Two grid searches pick the training hyperparameters of the paper's models. Both read checkpoints by
file name, so fits must follow the conventions below. Models are trained with the study runner
`weinhardt2026/studies/<study>/<study>.py` (or the cluster scripts in `slurm_jobs/`).

## SPICE — `run_spice_grid_search.py`

Fits: `slurm_jobs/grid_search.sh` → `weinhardt2026/run.py`, one checkpoint per grid point and seed, in
`weinhardt2026/studies/<study>/params/params_grid_search/`:

    spice_<study>_th<pruning_threshold>_sw<sindy_weight>_al<sindy_alpha>_er<ensemble_ratio>_gp<gate_penalty>_fp<feature_penalty>_rs<seed>.pkl

Every `<tag><value>` segment becomes a column (`HP_TAGS`); checkpoints with fewer tags than the
others (older grids) and `stage1` checkpoints are skipped. Each checkpoint is scored by
`evaluate_checkpoint` (in-sample AIC/BIC and trial likelihood, held-out trial likelihood, RNN-only
reference). Output: `weinhardt2026/studies/<study>/results/grid_search/`.

## Benchmark and GRU — `benchmark_grid_search.py`, `run_benchmark_grid_search.py`

Fits: `slurm_jobs/benchmark_grid_search.sh` → `benchmark_grid_search.py` (one grid point, all seeds),
saved with their config and losses in `runs/<study>/`:

    benchmark_<study>_lr<lr>_sc<scheduler 0/1>_rs<seed>.pkl / .json
    gru_<study>_lr<lr>_sc<scheduler 0/1>_hs<hidden_size>_do<dropout>_rs<seed>.pkl / .json

`run_benchmark_grid_search.py` aggregates the test loss over seeds (median) per grid point and selects
the grid point with the lowest median per study and model kind. Output:
`weinhardt2026/results/benchmark_grid_search/` (`grid_search_configs.csv`, `grid_search_best.csv`,
`figure_benchmark_grid_search`).

## Final models

The selected checkpoints of all seeds are copied to `weinhardt2026/studies/<study>/params/` under the
same names (`spice_…`, `benchmark_…`, `gru_…`); `weinhardt2026/utils/final_models.py` reads them from
there for every other analysis.
