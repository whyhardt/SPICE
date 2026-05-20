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
#SBATCH --array=0-13
#SBATCH --output=slurm_out/spice_castro2025_v7_sweep_%a.out

# ============================================================================
# v7 sweep: module × epochs_warmup × sindy_weight
#
# ID | Module | WU  | SW   | Suffix
# ---|--------|-----|------|------------------
#  0 | v2     | -   | 0    | v2_sw0
#  1 | v2     | 250 | 0.01 | v2_wu250_sw001
#  2 | v2     | 250 | 0.1  | v2_wu250_sw01
#  3 | v2     | 250 | 1    | v2_wu250_sw1
#  4 | v2     | 500 | 0.01 | v2_wu500_sw001
#  5 | v2     | 500 | 0.1  | v2_wu500_sw01
#  6 | v2     | 500 | 1    | v2_wu500_sw1
#  7 | v3     | -   | 0    | v3_sw0
#  8 | v3     | 250 | 0.01 | v3_wu250_sw001
#  9 | v3     | 250 | 0.1  | v3_wu250_sw01
# 10 | v3     | 250 | 1    | v3_wu250_sw1
# 11 | v3     | 500 | 0.01 | v3_wu500_sw001
# 12 | v3     | 500 | 0.1  | v3_wu500_sw01
# 13 | v3     | 500 | 1    | v3_wu500_sw1
#
# SW=0 jobs: pure RNN baseline (warmup irrelevant, one per module)
# ============================================================================

# Setup inductor and cache
JOB_UID="${SLURM_JOB_ID}_$(date +%s%N)"
export TORCHINDUCTOR_CACHE_DIR=/share/users/staff/d/dweinhardt/torchinductor_${JOB_UID}
export TRITON_CACHE_DIR=/share/users/staff/d/dweinhardt/triton_cache_${JOB_UID}
export TMPDIR=/share/users/staff/d/dweinhardt/tmp_${JOB_UID}
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR

# Job parameters indexed by SLURM_ARRAY_TASK_ID
MODULES=(
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_2
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
    studies.castro2025.spice_castro2025_3
)

SINDY_WEIGHTS=(0 0.01 0.1 1  0.01 0.1 1  0 0.01 0.1 1  0.01 0.1 1)
WARMUPS=(      0 250  250 250 500  500 500 0 250  250 250 500  500 500)

SUFFIXES=(
    v2_sw0
    v2_wu250_sw001
    v2_wu250_sw01
    v2_wu250_sw1
    v2_wu500_sw001
    v2_wu500_sw01
    v2_wu500_sw1
    v3_sw0
    v3_wu250_sw001
    v3_wu250_sw01
    v3_wu250_sw1
    v3_wu500_sw001
    v3_wu500_sw01
    v3_wu500_sw1
)

ID=$SLURM_ARRAY_TASK_ID

cd repos/SPICE
source activate spice
python weinhardt2026/run.py \
    --model weinhardt2026/studies/castro2025/params/spice_castro2025_${SUFFIXES[$ID]}.pkl \
    --data weinhardt2026/studies/castro2025/data/eckstein2024.csv \
    --test_sessions 2 \
    --results \
    --module ${MODULES[$ID]} \
    --sindy_weight ${SINDY_WEIGHTS[$ID]} \
    --epochs_warmup ${WARMUPS[$ID]} \
