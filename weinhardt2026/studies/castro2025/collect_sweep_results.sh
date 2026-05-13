#!/bin/bash
# Collect sweep results from slurm output files
# Usage: bash collect_sweep_results.sh

OUTDIR=~/slurm_out

printf "%-35s %10s %10s %10s %10s\n" "Config" "RNN-Train" "RNN-Test" "SPICE-Trn" "SPICE-Tst"
printf "%-35s %10s %10s %10s %10s\n" "-----------------------------------" "----------" "----------" "----------" "----------"

for f in "$OUTDIR"/spice_castro2025_62_sweep_*.out; do
    [ -f "$f" ] || continue
    id=$(echo "$f" | grep -oP 'sweep_\K\d+')

    # Extract Trial Lik. values from the two evaluation blocks
    # Train block: lines after "Evaluation on train data:"
    # Test block: lines after "Evaluation on test data:"
    train_rnn=$(awk '/Evaluation on train data:/,/Evaluation on test data/{if(/SPICE-RNN/) print $2}' "$f")
    train_spice=$(awk '/Evaluation on train data:/,/Evaluation on test data/{if(/SPICE  /) print $2}' "$f")
    test_rnn=$(awk '/Evaluation on test data:/,/^$/{if(/SPICE-RNN/) print $2}' "$f")
    test_spice=$(awk '/Evaluation on test data:/,/^$/{if(/SPICE  /) print $2}' "$f")

    # Get config suffix from the model save path
    suffix=$(grep -oP 'spice_castro2025_62_\K[^.]+' "$f" | head -1)
    [ -z "$suffix" ] && suffix="job_$id"

    printf "%-35s %10s %10s %10s %10s\n" "$suffix" "$train_rnn" "$test_rnn" "$train_spice" "$test_spice"
done | sort
