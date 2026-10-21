# Build against Pythia 8.315 from CERN LCG_108 on cvmfs (ACCRE):
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
g++ -O2 main214.cc -o generate $(pythia8-config --cxxflags --ldflags)
# Run: ./generate [nEvent] [seed] [outPrefix]
