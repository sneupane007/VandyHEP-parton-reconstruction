#!/bin/bash
# Score the four Sherpa cross-test models (trained on Ahadic or Lund, current loss or mass
# penalty 0.03) on each held-out test set: Sherpa Ahadic, Sherpa Lund, and Pythia
# (data/veto_test.npz). Writes results/crosstest_<test>.{txt,png} and per-model jet-mass
# plots in results/crosstest_mass/. Submitted by after_generation.sh to run once the four
# trainings end; can also be run by hand (repo root): bash sherpa/evaluate_crosstest.sh
set -e
cd "$(dirname "$0")/.."
PY="env -i HOME=$HOME PATH=/usr/bin:/bin OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} .venv/bin/python"
run() { ls -d checkpoints/gnn_attnrel_h128_k24_sherpa_$1_2* | tail -1; }
RUNS=("Ahadic-trained=$(run ahadic)" "Ahadic-trained + mass 0.03=$(run ahadic_m0.03)"
      "Lund-trained=$(run lund)" "Lund-trained + mass 0.03=$(run lund_m0.03)")
mkdir -p results/crosstest_mass
for test in ahadic lund pythia; do
  [ $test = pythia ] && T=data/veto_test.npz || T=data/sherpa_${test}_test.npz
  echo "=== test set: $test ($T)"
  $PY repro/evaluate_test.py $T results/crosstest_$test "${RUNS[@]}" 2>&1 | grep -iv warn || true
  for spec in "${RUNS[@]}"; do
    label=${spec%%=*}; dir=${spec#*=}
    tag=$(echo "$label" | tr -d ' .+' | tr 'A-Z' 'a-z')
    $PY repro/jet_mass_plot.py $dir/checkpoint_best.pt $T --all-events \
      --label "$label, tested on $test" --out results/crosstest_mass/${tag}_on_$test.png 2>&1 | grep -E "pred vs|saved" || true
  done
done
