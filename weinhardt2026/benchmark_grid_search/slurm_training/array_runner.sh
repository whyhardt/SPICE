#!/bin/bash
#SBATCH --job-name=spice_sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/staff/i/isdelpozo/SPICE/weinhardt2026/slurm_training/slurm_out/weber2024/%A_%a.out
#SBATCH --error=/home/staff/i/isdelpozo/SPICE/weinhardt2026/slurm_training/slurm_out/weber2024/%A_%a.err
#SBATCH --array=0-159%5 #%2 limits to only run 5 at once, rest queued
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "STARTING SCRIPT"
pwd
set -eo pipefail #set -euo pipefail # more strict error handling

# The iteration is determined by the SLURM array task ID, unless cfg/replica are set explicitly.
ITERATION=${SLURM_ARRAY_TASK_ID}

export TORCHINDUCTOR_CACHE_DIR=/home/staff/i/isdelpozo/torchinductor
export TRITON_CACHE_DIR=/home/staff/i/isdelpozo/triton_cache
export TMPDIR=/home/staff/i/isdelpozo/tmp

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

echo "=========================================="
echo "Starting gru+benchmark training for iteration ${ITERATION}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spice

# Navigate to project dir
cd /home/staff/i/isdelpozo
export PYTHONPATH=/home/staff/i/isdelpozo/SPICE:$PYTHONPATH

# Run one entry from the run table, or a specific cfg_id/replica pair.
RUN_TABLE="SPICE/weinhardt2026/slurm_training/json_outputs/weber2024_runs.jsonl"
OUT_DIR="SPICE/weinhardt2026/slurm_training/runs/weber2024"

# Example for rerunning one exact instance:
# CFG_ID="gru_weber2024_abcdef1234"
# REPLICA=7

if [ -n "${CFG_ID:-}" ]; then
	python SPICE/weinhardt2026/slurm_training/runner.py \
		--run-table "$RUN_TABLE" \
		--cfg_id "$CFG_ID" \
		${REPLICA:+--replica "$REPLICA"} \
		--out-dir "$OUT_DIR"
else
	python SPICE/weinhardt2026/slurm_training/runner.py \
		--run-table "$RUN_TABLE" \
		--index "$ITERATION" \
		--out-dir "$OUT_DIR"
fi

# Remove empty error file
ERR_FILE="/home/staff/i/isdelpozo/SPICE/weinhardt2026/slurm_training/slurm_out/weber2024/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
fi

echo "=========================================="
echo "Training completed for iteration ${ITERATION}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="