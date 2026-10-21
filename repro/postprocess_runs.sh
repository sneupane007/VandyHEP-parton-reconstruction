#!/bin/bash
# Poll checkpoints/; when a run has config.json (written at the end of training) and no
# postprocess.txt, make its plots and energy check. Exits once N runs (default 4) are done.
cd "$(dirname "$0")/.."
N=${1:-4}
while :; do
  for cfg in checkpoints/*/config.json; do
    [ -e "$cfg" ] || continue
    run=$(dirname $cfg); [ -s $run/postprocess.txt ] && continue
    pre=$(grep -o '"hadron_json": "[^"]*' $cfg | sed 's/.*": "//; s/_hadron_level_graphs_.json//; s|.*/data/|data/|')
    { echo "=== $run ($pre)"; bash repro/make_plots.sh $run/checkpoint_best.pt $pre
      env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python repro/energy_check.py $run/checkpoint_best.pt $pre; } 2>&1 | grep -iv warn > $run/postprocess.txt
    echo "$(date +%T) processed $run"
  done
  [ $(ls checkpoints/*/postprocess.txt 2>/dev/null | wc -l) -ge $N ] && break
  sleep 120
done
