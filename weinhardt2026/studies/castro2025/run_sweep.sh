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
#SBATCH --output=slurm_out/spice_castro2025_62_sweep_%a.out

# ============================================================================
# v62 sweep: scheduler × ensemble × pruning_method × sindy_weight
#
# ID | Sched | E  | Method | SW   | Suffix
# ---|-------|----|--------|------|--------------------------
#  0 | on    |  1 | ci     | 0.01 | schedon_e1_sw001
#  1 | on    |  1 | ci     | 0.1  | schedon_e1_sw01
#  2 | off   |  1 | ci     | 0.01 | schedoff_e1_sw001
#  3 | off   |  1 | ci     | 0.1  | schedoff_e1_sw01
#  4 | on    | 10 | ci     | 0.01 | schedon_e10_ci_sw001
#  5 | on    | 10 | ci     | 0.1  | schedon_e10_ci_sw01
#  6 | on    | 10 | ratio  | 0.01 | schedon_e10_ratio_sw001
#  7 | on    | 10 | ratio  | 0.1  | schedon_e10_ratio_sw01
#  8 | off   | 10 | ci     | 0.01 | schedoff_e10_ci_sw001
#  9 | off   | 10 | ci     | 0.1  | schedoff_e10_ci_sw01
# 10 | off   | 10 | ratio  | 0.01 | schedoff_e10_ratio_sw001
# 11 | off   | 10 | ratio  | 0.1  | schedoff_e10_ratio_sw01
#
# E=1 jobs: pruning method is irrelevant (ensemble pruning requires E>1)
# ============================================================================

# Setup inductor and cache
JOB_UID="${SLURM_JOB_ID}_$(date +%s%N)"
export TORCHINDUCTOR_CACHE_DIR=/share/users/staff/d/dweinhardt/torchinductor_${JOB_UID}
export TRITON_CACHE_DIR=/share/users/staff/d/dweinhardt/triton_cache_${JOB_UID}
export TMPDIR=/share/users/staff/d/dweinhardt/tmp_${JOB_UID}
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR

# Job parameters indexed by SLURM_ARRAY_TASK_ID
ENSEMBLES=(    1     1    1     1   10    10   10    10   10    10   10    10)
SINDY_WEIGHTS=(0.01  0.1  0.01  0.1 0.01  0.1  0.01  0.1  0.01  0.1  0.01  0.1)
LRS=(          0.001 0.001 0.01 0.01 0.001 0.001 0.001 0.001 0.01 0.01 0.01  0.01)
WARMUP_FACTORS=(10   10   1    1   10   10   10   10   1    1    1    1)
BOOST_SINDYS=( 10   10   1    1   10   10   10   10   1    1    1    1)
PRUNING_METHODS=(ci ci   ci   ci  ci   ci   ratio ratio ci  ci   ratio ratio)
PRUNING_TESTS=(0.05 0.05 0.05 0.05 0.05 0.05 0.7 0.7  0.05 0.05 0.7  0.7)
SUFFIXES=(
    schedon_e1_sw001
    schedon_e1_sw01
    schedoff_e1_sw001
    schedoff_e1_sw01
    schedon_e10_ci_sw001
    schedon_e10_ci_sw01
    schedon_e10_ratio_sw001
    schedon_e10_ratio_sw01
    schedoff_e10_ci_sw001
    schedoff_e10_ci_sw01
    schedoff_e10_ratio_sw001
    schedoff_e10_ratio_sw01
)

ID=$SLURM_ARRAY_TASK_ID

cd repos/SPICE
source activate spice
python weinhardt2026/run.py \
    --model weinhardt2026/studies/castro2025/params/spice_castro2025_62_${SUFFIXES[$ID]}.pkl \
    --data weinhardt2026/studies/castro2025/data/eckstein2024.csv \
    --test_sessions 2 \
    --results \
    --module studies.castro2025.spice_castro2025_62 \
    --sindy_weight ${SINDY_WEIGHTS[$ID]} \
    --epochs 1000 \
    --epochs_warmup 500 \
    --lr ${LRS[$ID]} \
    --pruning_frequency 100 \
    --pruning_method ${PRUNING_METHODS[$ID]} \
    --pruning_test ${PRUNING_TESTS[$ID]} \
    --ensemble ${ENSEMBLES[$ID]} \
    --embedding 8 \
    --lr_warmup_factor ${WARMUP_FACTORS[$ID]} \
    --lr_boost_sindy ${BOOST_SINDYS[$ID]}
