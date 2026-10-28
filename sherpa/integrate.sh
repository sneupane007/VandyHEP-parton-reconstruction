#!/bin/bash
# One-off: integrate the gamma+jet matrix element and keep Process/ + Results.zip in
# sherpa/integration/, so the generation tasks start producing events immediately.
# Both runcards share the hard process, so one integration serves Ahadic and Lund.
# Usage (repo root): bash sherpa/integrate.sh
set -e
cd "$(dirname "$0")"
source /data/rke_group/Software/Sherpa/setup_sherpa3.sh
rm -rf integration && mkdir integration && cd integration
cp ../ahadic.yaml Sherpa.yaml
Sherpa -e 0 > integrate.log 2>&1
rm Sherpa.yaml
ls
