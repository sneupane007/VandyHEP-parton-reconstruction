#!/bin/bash
#SBATCH --job-name=pr_genveto
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --array=1-100
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=data/raw_veto/logs/gen_%a.out
# Pythia photon+jet, 14 TeV, prompt photon vetoed, hadron jet dR-matched to the leading parton jet (800-820 GeV).
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
cd data/raw_veto
SEED=$((100 + SLURM_ARRAY_TASK_ID))
../../parton_recon/data/generate/generate 10000 ${SEED} seed$(printf %03d ${SEED}) 1 > /dev/null
