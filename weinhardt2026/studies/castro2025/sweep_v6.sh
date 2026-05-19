#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres gpu
#SBATCH --constraint="A100|H100.80gb|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --output=slurm_out/spice_castro2025_v6_sweep_%a.out

# ============================================================================
# v6 architecture sweep: all model variants with fixed training config
#
# ID | File                         | LA  | Choice  | Spatial  | Exploration
# ---|------------------------------|-----|---------|----------|------------
#  0 | spice_castro2025_bias_5      | no  | split   | separate | split
#  1 | spice_castro2025_2           | yes | merged  | absorbed | split
#  2 | spice_castro2025_v6a         | yes | split   | separate | split
#  3 | spice_castro2025_v6b         | no  | split   | separate | chosen_only
#  4 | spice_castro2025_v6c         | yes | split   | separate | chosen_only
#  5 | spice_castro2025_v6d         | no  | split   | none     | split
#  6 | spice_castro2025_v6e         | yes | split   | none     | split
#  7 | spice_castro2025_v6f         | no  | merged  | absorbed | split
#  8 | spice_castro2025_v6g         | no  | merged  | absorbed | chosen_only
#  9 | spice_castro2025_v6h         | yes | merged  | absorbed | chosen_only
# 10 | spice_castro2025_v6i         | no  | split   | none     | chosen_only
# 11 | spice_castro2025_v6j         | yes | split   | none     | chosen_only
# ============================================================================

# Setup inductor and cache
JOB_UID="${SLURM_JOB_ID}_$(date +%s%N)"
export TORCHINDUCTOR_CACHE_DIR=/share/users/staff/d/dweinhardt/torchinductor_${JOB_UID}
export TRITON_CACHE_DIR=/share/users/staff/d/dweinhardt/triton_cache_${JOB_UID}
export TMPDIR=/share/users/staff/d/dweinhardt/tmp_${JOB_UID}
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR

MODULES=(
    studies.castro2025.spice_castro2025_bias_5
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_v6a
    studies.castro2025.spice_castro2025_v6b
    studies.castro2025.spice_castro2025_v6c
    studies.castro2025.spice_castro2025_v6d
    studies.castro2025.spice_castro2025_v6e
    studies.castro2025.spice_castro2025_v6f
    studies.castro2025.spice_castro2025_v6g
    studies.castro2025.spice_castro2025_v6h
    studies.castro2025.spice_castro2025_v6i
    studies.castro2025.spice_castro2025_v6j
)

NAMES=(
    spice_castro2025_bias_5
    spice_castro2025_2
    spice_castro2025_v6a
    spice_castro2025_v6b
    spice_castro2025_v6c
    spice_castro2025_v6d
    spice_castro2025_v6e
    spice_castro2025_v6f
    spice_castro2025_v6g
    spice_castro2025_v6h
    spice_castro2025_v6i
    spice_castro2025_v6j
)

ID=$SLURM_ARRAY_TASK_ID

cd repos/SPICE
source activate spice
python weinhardt2026/run.py \
    --data weinhardt2026/studies/castro2025/data/eckstein2024.csv \
    --module ${MODULES[$ID]} \
    --model weinhardt2026/studies/castro2025/params/${NAMES[$ID]}.pkl \
    --test_sessions 2 \
    --results
