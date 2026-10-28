#!/bin/bash
#SBATCH --job-name=pr_convert
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Vetoed sample -> compact arrays. Seeds 191-200 are a held-out test set no model trains on.
PY="env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python"
$PY scripts/convert_to_arrays.py data/raw_veto data/veto_train.npz --seeds 101-190
$PY scripts/convert_to_arrays.py data/raw_veto data/veto_test.npz --seeds 191-200
