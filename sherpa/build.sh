#!/bin/bash
# Build sherpa_jets (HepMC3 + FastJet from LCG_108 on cvmfs). Usage (repo root): bash sherpa/build.sh
set -e
cd "$(dirname "$0")"
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh > /dev/null 2>&1
g++ -O2 -std=c++17 sherpa_jets.cc -o sherpa_jets $(HepMC3-config --cflags --libs) $(fastjet-config --cxxflags --libs)
echo "built sherpa/sherpa_jets"
