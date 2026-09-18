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
#SBATCH --array=0-9
#SBATCH --job-name=spice
#SBATCH --output=slurm_out/%x_%a.out

# --------------------------------------------------------------------------
# Stability sweep for a single study (10 runs with default parameters).
#
# One instance per study — pick the study either by editing STUDY below or,
# without touching the file, at submission time:
#
#   sbatch --job-name=spice_kolff2025_groom --export=ALL,STUDY=kolff2025_groom slurm_spice_study.sh
#
# Optional overrides (same mechanism): N_RUNS, EPOCHS, ENSEMBLE, TAG.
# --------------------------------------------------------------------------

STUDY="${STUDY:-kolff2025_groom}"
N_RUNS="${N_RUNS:-10}"
TAG="${TAG:-stability}"

# setup of inductor and cache
JOB_UID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%s%N)"
export TORCHINDUCTOR_CACHE_DIR=/share/users/staff/d/dweinhardt/torchinductor_${JOB_UID}
export TRITON_CACHE_DIR=/share/users/staff/d/dweinhardt/triton_cache_${JOB_UID}
export TMPDIR=/share/users/staff/d/dweinhardt/tmp_${JOB_UID}
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $TMPDIR

cd repos/SPICE
source activate spice

# --------------------------------------------------------------------------
# Per-study configuration
#
# MODULE        module holding SpiceModel + CONFIG (importable with weinhardt2026 on the path)
# DATA          behavioral CSV
# TEST_BLOCKS   comma-separated held-out blocks (empty string = no split)
# MODEL_KWARGS  JSON kwargs forwarded to the SpiceModel constructor
# DATA_KWARGS   JSON kwargs forwarded to csv_to_dataset
# EXTRA_ARGS    anything else (e.g. --n_items, --epochs) as a bash array
# --------------------------------------------------------------------------

STUDY_DIR="weinhardt2026/studies/${STUDY}"
MODEL_KWARGS='{}'
DATA_KWARGS='{}'
EXTRA_ARGS=()

case "$STUDY" in

  dezfouli2019)
    MODULE="studies.dezfouli2019.spice_dezfouli2019"
    DATA="${STUDY_DIR}/data/dezfouli2019.csv"
    TEST_BLOCKS="3,6,9"
    MODEL_KWARGS='{"reward_binary": true}'
    ;;

  eckstein2026)
    MODULE="studies.eckstein2026.spice_eckstein2026"
    DATA="${STUDY_DIR}/data/eckstein2026.csv"
    TEST_BLOCKS="3"
    ;;

  eckstein2026_choice)
    # choice-only variant of the same dataset
    STUDY_DIR="weinhardt2026/studies/eckstein2026"
    MODULE="studies.eckstein2026.spice_eckstein2026_choice"
    DATA="${STUDY_DIR}/data/eckstein2026.csv"
    TEST_BLOCKS="3"
    ;;

  braun2018)
    MODULE="studies.braun2018.spice_braun2018"
    DATA="${STUDY_DIR}/data/braun2018.csv"
    TEST_BLOCKS="3,6,9"
    DATA_KWARGS='{"df_participant_id": "subject", "df_choice": "transcode", "df_feedback": null, "df_block": "block", "additional_inputs": ["difference", "current", "other"], "timeshift_additional_inputs": [-1, -1, -1]}'
    ;;

  bustamante2023)
    MODULE="studies.bustamante2023.spice_bustamante2023"
    DATA="${STUDY_DIR}/data/bustamante2023.csv"
    TEST_BLOCKS="3,6"
    DATA_KWARGS='{"df_participant_id": "subject_id", "df_choice": "decision", "df_feedback": "reward", "df_block": "overall_round", "additional_inputs": ["harvest_duration", "travel_duration"]}'
    ;;

  huang2026)
    MODULE="studies.huang2026.spice_huang2026"
    DATA="${STUDY_DIR}/data/huang2026.csv"
    TEST_BLOCKS="3,6,9,12,15,18,21,24,27,30,33,36,39,42"
    DATA_KWARGS='{"df_participant_id": "subject_ID", "df_block": "currentRound", "df_choice": "hover_tile_index", "df_feedback": null, "additional_inputs": ["partner_tile_index", "time_point"]}'
    MODEL_KWARGS='{"grid_size": 4, "time_max": 10}'
    ;;

  kolff2025_groom)
    STUDY_DIR="weinhardt2026/studies/kolff2025"
    MODULE="studies.kolff2025.spice_kolff2025_groom"
    # kolff2025_groom.csv is the perspective dataframe as get_dataset() saves it:
    # already filtered to bouts >= 5 events and with rank_diff_centered computed.
    DATA="${STUDY_DIR}/data/kolff2025_groom.csv"
    # The 56 held-out interactions of get_dataset(test_fraction=0.2, seed=42) — the same
    # split the notebook and the benchmark models use. Regenerate after a data change with:
    #   interactions = np.sort(df['interaction_id'].unique())
    #   np.random.default_rng(42).choice(interactions, size=56, replace=False)
    TEST_BLOCKS="13,17,19,20,21,24,25,35,52,55,57,64,69,85,104,108,113,114,115,125,128,131,137,139,140,147,155,156,160,166,180,181,186,191,194,201,205,208,210,217,220,224,226,230,231,236,239,246,254,255,263,268,270,289,293,307"
    DATA_KWARGS='{"df_participant_id": "focal", "df_block": "interaction_id", "df_choice": "outcome", "df_feedback": null, "additional_inputs": ["own_action", "partner_action", "rank_own", "rank_partner", "rank_diff", "rank_diff_centered"]}'
    ;;

  ganesh2024a)
    # 'contrast_difference_next' is derived by spice_ganesh2024a.prepare_dataset
    MODULE="studies.ganesh2024a.spice_ganesh2024a"
    DATA="${STUDY_DIR}/data/ganesh2024a_choice.csv"
    TEST_BLOCKS="3,6,9"
    DATA_KWARGS='{"df_participant_id": "subjID", "df_choice": "choice", "df_feedback": "reward", "df_block": "blocks", "additional_inputs": ["contrast_difference"]}'
    ;;

  weber2024)
    # sin/cos conversion + packed ys come from the hooks in spice_weber2024
    MODULE="studies.weber2024.spice_weber2024"
    DATA="${STUDY_DIR}/data/weber2024.csv"
    TEST_BLOCKS="2,5,9,14"
    DATA_KWARGS='{"df_participant_id": "participant", "df_experiment_id": "experiment", "df_choice": ["shield_sin", "shield_cos"], "df_feedback": ["laser_sin", "laser_cos"], "df_block": "block", "additional_inputs": ["laser_caught", "volatility", "stochasticity", "trial_duration_frames", "trueMean"], "continuous_action": true}'
    ;;

  bruckner2025)
    # 'z_next', position normalization and mu_t masking come from spice_bruckner2025.prepare_dataframe
    MODULE="studies.bruckner2025.spice_bruckner2025"
    DATA="${STUDY_DIR}/data/bruckner2025.csv"
    TEST_BLOCKS="1,2"
    DATA_KWARGS='{"df_choice": "b_t", "df_feedback": "x_t", "additional_inputs": ["z_next", "catch", "v_t", "sigma", "r_t", "mu_t", "c_t"], "continuous_action": true, "timeshift_additional_inputs": [0, 0, 0, 0, 0, 0, 0]}'
    ;;

  *)
    echo "ERROR: unknown study '$STUDY'." >&2
    echo "       Known: dezfouli2019 eckstein2026 eckstein2026_choice braun2018 bustamante2023" >&2
    echo "              ganesh2024a weber2024 bruckner2025 huang2026 kolff2025_groom" >&2
    exit 1
    ;;
esac

# --------------------------------------------------------------------------
# Stability runs with default parameters
# --------------------------------------------------------------------------

RUN_IDX=$(( SLURM_ARRAY_TASK_ID % N_RUNS ))
MODEL="${STUDY_DIR}/params/spice_${STUDY}_${TAG}_${RUN_IDX}.pkl"
mkdir -p "$(dirname "$MODEL")" slurm_out

[ -n "$EPOCHS" ] && EXTRA_ARGS+=(--epochs "$EPOCHS")
[ -n "$ENSEMBLE" ] && EXTRA_ARGS+=(--ensemble "$ENSEMBLE")
[ -n "$TEST_BLOCKS" ] && EXTRA_ARGS+=(--test_blocks "$TEST_BLOCKS")

echo "Stability run — study=$STUDY task=$SLURM_ARRAY_TASK_ID run=$RUN_IDX"
echo "  module: $MODULE"
echo "  data:   $DATA"
echo "  model:  $MODEL"
echo "  blocks: ${TEST_BLOCKS:-<no split>}"

python weinhardt2026/run.py \
--module "$MODULE" \
--data "$DATA" \
--model "$MODEL" \
--model_kwargs "$MODEL_KWARGS" \
--data_kwargs "$DATA_KWARGS" \
--results \
"${EXTRA_ARGS[@]}"
