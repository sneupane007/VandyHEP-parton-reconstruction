#!/bin/bash
#SBATCH --job-name=pr_train_repro
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Faithful reproduction of the Oct-13 chamfer + LogPt run (1000 epochs, batch 32, seed 0)
# on 6725 events (-> 1345 val, as in emd_distribution.png).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python -u repro/train_oct13.py \
  --hadron-json data/repro6725_hadron_level_graphs_.json \
  --parton-json data/repro6725_parton_level_graphs_.json \
  --epochs 1000 --batch-size 32 --seed 0 "$@"
