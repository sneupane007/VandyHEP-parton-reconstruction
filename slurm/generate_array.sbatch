#!/bin/bash
#SBATCH --job-name=pr_gen
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --array=1-100
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=data/raw/logs/gen_%a.out
# Pythia photon+jet, 14 TeV, leading parton jet 800-820 GeV (main214.cc). One seed per task.
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd data/raw
../../parton_recon/data/generate/generate 10000 ${SLURM_ARRAY_TASK_ID} seed$(printf %03d ${SLURM_ARRAY_TASK_ID}) > /dev/null
