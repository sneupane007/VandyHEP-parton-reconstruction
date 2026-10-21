#!/bin/bash
# Make emd_distribution.png and pca_corner.png (written next to the checkpoint) for a LogPt
# checkpoint. Usage: bash repro/make_plots.sh checkpoints/<run>/checkpoint_best.pt [data prefix]
# The data prefix must be the dataset the checkpoint was trained on (same seed-0 val split).
set -e
CKPT=$1
PREFIX=${2:-data/repro6725}
DATA="--hadron-json ${PREFIX}_hadron_level_graphs_.json --parton-json ${PREFIX}_parton_level_graphs_.json"
PY="env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python"
$PY repro/evaluate_oct13.py "$CKPT" $DATA
$PY scripts/pca_compare_model.py "$CKPT" $DATA
