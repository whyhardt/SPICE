# Paper results

All figures and numbers of the paper, produced from the final models (`studies/<study>/params/`, 10 seeds per model
kind; see `weinhardt2026/utils/final_models.py`) by the paper runners in `weinhardt2026/analysis/`. Commands run from
the repository root; details in `docs/analyses.md`. Every figure exists annotated (`<name>.pdf/.png`) and clean
(`<name>_clean.pdf/.png`), next to the CSV with its numbers.

## Across studies — `weinhardt2026/results/`

| Item | Folder | Produced by |
|---|---|---|
| Model evaluation | `model_evaluation/` — figure, `model_evaluation_summary.csv` (mean/SD over seeds: held-out prediction, PPC, AIC, BIC, parameters) | `python weinhardt2026/analysis/evaluation/run_model_evaluation.py` |
| Parameter recovery | `parameter_recovery/` — `*_metrics_by_mechanism.csv` (precision, recall, F1 per dataset size × ground-truth model), figures, `*_metrics.npz` | `python weinhardt2026/analysis/validation/run_parameter_recovery.py` |
| Model identification | `model_identification/` — confusion matrices SPICE vs. benchmark (held-out loglik, AIC, BIC) | `python weinhardt2026/analysis/validation/model_identification/run_model_identification.py` (after the pipeline in that folder) |
| Benchmark / GRU grid search | `benchmark_grid_search/` — `grid_search_configs.csv`, `grid_search_best.csv`, figure (selected grid point in red) | `python weinhardt2026/analysis/grid_search/run_benchmark_grid_search.py` |

## Per study — `weinhardt2026/studies/<study>/results/`

| Item | Folder | Produced by |
|---|---|---|
| Model evaluation | `model_evaluation/model_evaluation.csv` (per model and seed) | `run_model_evaluation.py` |
| Equation discovery | `equation_discovery/` — mechanisms (term sets grouped together in ≥ 6 of 10 seeds): reduced equation matrix (`figure_equation_matrix`), correlations between the mechanisms (`figure_mechanism_correlations` + `.csv`), joint-support dendrogram (`figure_mechanism_dendrogram`, `mechanisms.csv`), beta effects per criterion (`figure_mechanism_betas_<contrast>`, `betas_structural_mechanisms_<contrast>.csv`) | `python weinhardt2026/analysis/discovery/run_equation_discovery.py [<study>]` (not kolff2025: group model; criteria: `CRITERIA`) |
| Equation stability | `equation_stability/` — agreement across seeds per term, coefficients per seed | `python weinhardt2026/analysis/validation/run_equation_stability.py all` |
| Posterior predictive checks | `posterior_predictive/` — `ppc_*.png/.pdf` (nRMSE per model in every panel), `ppc_scores*.csv` | `python weinhardt2026/analysis/evaluation/run_posterior_predictive.py all` |
| Morphing (dezfouli2019) | `morphing/` — equations along a behavioral axis | `python weinhardt2026/analysis/explorative/run_morphing.py` (default metric: reward rate); cluster: `sbatch slurm_jobs/morphing.sh` |
| SPICE grid search (dezfouli2019) | `grid_search/` | `python weinhardt2026/analysis/grid_search/run_spice_grid_search.py` |
