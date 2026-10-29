#!/bin/bash
#SBATCH --job-name=pr_sherpa_post
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Run after the generation array: sbatch --dependency=afterany:<array job id> sherpa/after_generation.sbatch
cd $SLURM_SUBMIT_DIR
sacct -n -X -j ${GEN_JOB:-0} -o state 2>/dev/null | sort | uniq -c
echo "jet files: $(ls data/raw_sherpa/*.txt | wc -l) of 200"
bash sherpa/after_generation.sh 2>&1 | grep -iv warn
