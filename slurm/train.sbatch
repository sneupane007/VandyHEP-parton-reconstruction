#!/bin/bash
#SBATCH --job-name=pr_train
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Usage: sbatch -J <name> slurm/train.sbatch <train script> <data prefix | file.npz> [extra args]
#   e.g. sbatch -J matched_veto slurm/train.sbatch scripts/train2.py data/veto6725
# 1000 epochs, batch 32, seed 0 -- same settings as the Oct-13 run.
SCRIPT=$1; PREFIX=$2; shift 2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "$SCRIPT on $PREFIX, job $SLURM_JOB_ID"
if [[ $PREFIX == *.npz ]]; then
  DATA="--hadron-json $PREFIX --parton-json $PREFIX"      # compact arrays hold both levels
else
  DATA="--hadron-json ${PREFIX}_hadron_level_graphs_.json --parton-json ${PREFIX}_parton_level_graphs_.json"
fi
env -i HOME=$HOME PATH=/usr/bin:/bin OMP_NUM_THREADS=$OMP_NUM_THREADS .venv/bin/python -u "$SCRIPT" \
  $DATA --epochs 1000 --batch-size 32 --seed 0 "$@"
