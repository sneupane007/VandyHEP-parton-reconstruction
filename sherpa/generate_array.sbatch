#!/bin/bash
#SBATCH --job-name=pr_sherpa
#SBATCH --account=rke_group
#SBATCH --partition=batch
#SBATCH --array=1-200
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=data/raw_sherpa/logs/%a.out
# Sherpa 3.0.1 gamma+jet (q g -> q gamma), 14 TeV, CSS shower, MPI on. Tasks 1-100: Ahadic
# (cluster) hadronization, seeds 1-100; tasks 101-200: Pythia8 Lund string, seeds 1-100.
# The runcards differ only in FRAGMENTATION. Events stream through a FIFO into sherpa_jets
# (main214.cc selection with photon veto and dR matching); nothing but jets is written.
set -e
R=$SLURM_SUBMIT_DIR
i=$SLURM_ARRAY_TASK_ID
if [ $i -le 100 ]; then frag=ahadic; seed=$i; else frag=lund; seed=$((i - 100)); fi
source /data/rke_group/Software/Sherpa/setup_sherpa3.sh > /dev/null
W=${TMPDIR:-/tmp}/sherpa_${SLURM_JOB_ID}_$i; mkdir -p $W; cd $W
cp $R/sherpa/$frag.yaml Sherpa.yaml
# Reuse the matrix-element integration (make it once with sherpa/integrate.sh); without it every
# task would integrate on its own first, which works but costs ~1 min per task.
[ -d $R/sherpa/integration/Process ] && cp -r $R/sherpa/integration/Process $R/sherpa/integration/Results.zip .
mkfifo events
OUT=$R/data/raw_sherpa/${frag}_seed$(printf %03d $seed).txt
$R/sherpa/sherpa_jets events $OUT.tmp 2> ana.log &
Sherpa -e 10000 -R $seed 'EVENT_OUTPUT: HepMC3_GenEvent[events]' > sherpa.log 2>&1
wait
mv $OUT.tmp $OUT
echo "$frag seed $seed: $(cat ana.log | grep sherpa_jets)"
rm -rf $W
