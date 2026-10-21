# Paper results

All figures and numbers of the paper, produced from the final models (`studies/<study>/params/`, 10 seeds per model
kind; see `weinhardt2026/utils/final_models.py`). Commands run from the repository root; details in `docs/analyses.md`.

## Across studies — `weinhardt2026/results/`

| Item | Folder | Produced by |
|---|---|---|
| Model evaluation | `model_evaluation/` — figure, `model_evaluation_summary.csv` (mean/SD over seeds: held-out prediction, PPC, AIC, BIC, parameters) | `python weinhardt2026/figures/figure_model_evaluation.py` |
| Parameter recovery | `parameter_recovery/` — `*_metrics_by_mechanism.csv` (precision, recall, F1 per dataset size × ground-truth model), figures, `*_metrics.npz` | `python weinhardt2026/analysis/analysis_parameter_recovery.py` |
| Model identification | `model_identification/` — confusion matrices SPICE vs. benchmark (held-out loglik, AIC, BIC) | `python weinhardt2026/confusion_matrix/analysis_confusion_matrix.py` |

## Per study — `weinhardt2026/studies/<study>/results/`

| Item | Folder | Produced by |
|---|---|---|
| Model evaluation | `model_evaluation/model_evaluation.csv` (per model and seed) | `figure_model_evaluation.py` |
| Discovered mechanisms | `mechanisms/` — term groups recurring across seeds, `core_mechanisms_distances_votes5.csv`; dezfouli2019: `trajectories/` | `python weinhardt2026/figures/figure_mechanisms.py --study <study>` (not kolff2025: group model) |
| Beta effects | `beta_effects/` — `betas_structural_<study>_mechanisms_votes5_<criterion>.csv/.png` | `figure_mechanisms.py` (criteria: `BETA_CRITERIA`; `--merged` adds merged groups, e.g. Patients vs Control) |
| Equation stability | `equation_stability/` — agreement across seeds per term, coefficients per seed | `python weinhardt2026/figures/figure_equation_stability.py all` |
| Posterior predictive checks | `posterior_predictive/` — `ppc_*.png/.pdf` (nRMSE per model in every panel), `ppc_scores*.csv` | `python weinhardt2026/analysis/analysis_posterior_predictive.py all` |
| Equation tables | `equations/` — `term_presence.csv` (seed-averaged presence per participant and term), `example_participants.csv/.txt` (two most distinct + most average participant in embedding space) | `python weinhardt2026/analysis/analysis_equation_tables.py all` |
| Morphing | `morphing/` — equations along a behavioral axis | `python weinhardt2026/analysis/analysis_morphing.py <study>` (default metric: reward rate); cluster: `sbatch --export=ALL,STUDY=<study> slurm_jobs/morphing.sh` |
| Hyperparameter grid search | `grid_search/` (dezfouli2019) | `analysis/analysis_grid_search.py`, `analysis/analysis_grid_mechanisms.py` |
