#!/bin/bash
# After sherpa/generate_array.sbatch: build train/test .npz per hadronization model, check
# that the two share the same parton level (same shower) and differ only at hadron level,
# then submit the cross-hadronization trainings (current loss and mass penalty 0.03, each
# on 6725 jets, as the Pythia loss tests), and the job that scores them when they end.
# Usage (repo root): bash sherpa/after_generation.sh
set -e
cd "$(dirname "$0")/.."
PY="env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python"
for frag in ahadic lund; do
  $PY sherpa/to_npz.py data/raw_sherpa $frag 1-90 data/sherpa_${frag}_train.npz | tail -1
  $PY sherpa/to_npz.py data/raw_sherpa $frag 91-100 data/sherpa_${frag}_test.npz | tail -1
done

$PY - <<'EOF'
import sys; sys.path.insert(0, ".")
import numpy as np
from parton_recon.data import JetPairDataset
from parton_recon.metrics import jet_mass
print(f"{'sample':22s} {'level':6s} {'mult':>6s} {'mass med':>9s} {'mass mean':>9s} {'lead z':>7s}")
for name, path in [("Sherpa Ahadic", "data/sherpa_ahadic_test.npz"), ("Sherpa Lund", "data/sherpa_lund_test.npz"),
                   ("Pythia (main214)", "data/veto_test.npz")]:
    d = JetPairDataset(path, None)
    for lvl, view in [("parton", d.parton), ("hadron", d.hadron)]:
        ev = [view.array(i) for i in range(len(d))]
        mult = np.array([len(e) for e in ev]); m = np.array([jet_mass(e) for e in ev])
        z = np.array([e[:, 0].max() / e[:, 0].sum() for e in ev])
        print(f"{name:22s} {lvl:6s} {np.median(mult):6.0f} {np.median(m):9.1f} {m.mean():9.1f} {np.median(z):7.3f}")
EOF

# SUBMIT=0: stop here (the trainings were already queued with a dependency on this step).
[ "${SUBMIT:-1}" = 0 ] && exit 0

C="slurm/train.sbatch scripts/train_factorized.py"
A="--model attention --epochs 300 --patience 40 --max-events 6725 --workers 2"
JOBS=()
for frag in ahadic lund; do
  JOBS+=($(sbatch --parsable -J x_${frag}      --mem=24G $C data/sherpa_${frag}_train.npz $A --tag _sherpa_${frag}))
  JOBS+=($(sbatch --parsable -J x_${frag}_m003 --mem=24G $C data/sherpa_${frag}_train.npz $A --mass-weight 0.03 --tag _sherpa_${frag}_m0.03))
done
echo "cross-test trainings: ${JOBS[*]}"
# Score all four on the Ahadic, Lund and Pythia test sets once they have all ended.
DEP=$(IFS=:; echo "${JOBS[*]}")
sbatch -J x_eval --dependency=afterany:$DEP --account=rke_group --partition=batch --cpus-per-task=8 --mem=24G \
  --time=06:00:00 --output=slurm/logs/%x_%j.out --wrap "bash sherpa/evaluate_crosstest.sh"
