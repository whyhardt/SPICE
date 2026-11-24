set -e
cd /home/daniel/repositories/SPICE
python -u weinhardt2026/analysis/evaluation/run_posterior_predictive.py weber2024 --recompute > logs/weber_ppc.log 2>&1
python -u weinhardt2026/analysis/evaluation/run_model_evaluation.py > logs/weber_model_eval.log 2>&1
python -u weinhardt2026/analysis/validation/run_equation_stability.py weber2024 > logs/weber_stability.log 2>&1
python -u weinhardt2026/analysis/discovery/run_equation_discovery.py weber2024 > logs/weber_discovery.log 2>&1
echo ALL DONE
