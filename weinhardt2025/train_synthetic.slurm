#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=spice
#SBATCH --output=slurm_out/synthetic_%a_%j.out
#SBATCH --error=slurm_out/synthetic_%a_%j.err
#SBATCH --array=0-7
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --constraint="A100|H100.80gb"

# The iteration is determined by the SLURM array task ID
ITERATION=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Starting SPICE training for iteration ${ITERATION}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Activate your conda environment (adjust as needed)
source activate spice
# or: module load anaconda && conda activate spice

# Navigate to the project directory
cd repos/SPICE

# Define participant counts to iterate over
PARTICIPANTS=(32 64 128 256 512)

# Loop over all participant counts for this iteration
for N_PARTICIPANTS in "${PARTICIPANTS[@]}"; do

    DATA_FILE="weinhardt2025/data/synthetic/synthetic_${N_PARTICIPANTS}p_${ITERATION}_0.csv"
    MODEL_FILE="weinhardt2025/params/synthetic/spice_synthetic_${N_PARTICIPANTS}p_${ITERATION}_0.pkl"

    echo ""
    echo "------------------------------------------"
    echo "Training on ${DATA_FILE}"
    echo "Model will be saved to ${MODEL_FILE}"
    echo "Started at: $(date)"
    echo "------------------------------------------"

    # Create params directory if it doesn't exist
    mkdir -p weinhardt2025/params/synthetic

    python weinhardt2025/run.py \
        --data "${DATA_FILE}" \
        --model "${MODEL_FILE}" \


    echo "Finished ${N_PARTICIPANTS} participants at: $(date)"

done

echo ""
echo "=========================================="
echo "All training completed for iteration ${ITERATION}"
echo "Finished at: $(date)"
echo "=========================================="
