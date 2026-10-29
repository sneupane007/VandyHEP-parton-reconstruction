#!/bin/bash
# Score the full-sample mass-penalty run against the full-sample current-loss run and the
# 6.7k mass-penalty run on the held-out Pythia test set, plus its jet-mass plot.
cd "$(dirname "$0")/.."
PY="env -i HOME=$HOME PATH=/usr/bin:/bin OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} .venv/bin/python"
NEW=$(ls -d checkpoints/gnn_attnrel_h128_k24_full_m0.03_2* | tail -1)
$PY repro/evaluate_test.py data/veto_test.npz results/test_full_mass \
  "Full 53.9k, current loss=checkpoints/gnn_attnrel_h128_k24_full_20261026_220935" \
  "Full 53.9k + mass 0.03=$NEW" \
  "6.7k + mass 0.03=checkpoints/gnn_attnrel_h128_k24_m0.03_20261026_204825" 2>&1 | grep -iv warn
$PY repro/jet_mass_plot.py $NEW/checkpoint_best.pt data/veto_test.npz --all-events \
  --label "+ mass penalty 0.03, 53.9k jets" --out results/jet_mass/full_54k_mass003.png 2>&1 | grep -E "pred vs|saved"
