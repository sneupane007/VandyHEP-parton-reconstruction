// Paired parton/hadron jets from Sherpa HepMC3 events, with the selection of main214.cc.
//
// Usage: sherpa_jets <hepmc3 file or fifo> <out.txt>
//
// Parton level: the particles entering the hadronization vertex, i.e. a vertex whose
// incoming particles are all partons (quarks, gluons, diquarks) and whose outgoing ones
// include hadrons. Same content as Pythia's isFinalPartonLevel(): shower, MPI and
// beam-remnant partons. Works for FRAGMENTATION: Ahadic (one such vertex) and Pythia8
// (Lund; Sherpa adds a second hadron -> hadron bookkeeping vertex, which this skips).
// Hadron level: status 1, no neutrinos, pT > 0.1 GeV. The prompt photon (a photon whose
// photon-only ancestry reaches the status-3 hard-process photon) is removed from the
// hadron level; it never enters hadronization, so it is not in the parton level.
// Both levels: |eta| < 2.5, anti-kT R = 0.8 (FastJet, E-scheme). The leading parton
// jet must have 800 < pT < 820 GeV; the hadron jet is the one closest to it in (y, phi)
// within dR < 0.4, among hadron jets with pT > 600 GeV. As main214.cc with vetoAndMatch=1.
//
// Output, one block per accepted event:
//   E <n_hadron> <n_parton>
//   <pT> <eta> <phi>      x n_hadron   (hadron-jet constituents)
//   <pT> <eta> <phi>      x n_parton   (parton-jet constituents)
// sherpa/to_npz.py turns this into the .npz format JetPairDataset reads.

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/ReaderAscii.h"
#include "fastjet/ClusterSequence.hh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

using namespace HepMC3;

static bool isParton(int pid) {
  int a = std::abs(pid);
  return a <= 6 || a == 21 || (a > 1000 && a < 6000 && (a / 10) % 10 == 0);  // quarks, gluons, diquarks
}

static bool isPromptPhoton(const ConstGenParticlePtr& p) {
  ConstGenParticlePtr cur = p;
  while (cur && cur->pid() == 22) {
    if (cur->status() == 3) return true;
    auto v = cur->production_vertex();
    if (!v) return false;
    ConstGenParticlePtr parent;
    for (const auto& q : v->particles_in())
      if (q->pid() == 22) { parent = q; break; }
    cur = parent;
  }
  return false;
}

static fastjet::PseudoJet toPJ(const ConstGenParticlePtr& p) {
  const auto& m = p->momentum();
  return fastjet::PseudoJet(m.px(), m.py(), m.pz(), m.e());
}

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: sherpa_jets <hepmc3> <out.txt>\n"; return 1; }
  ReaderAscii reader(argv[1]);
  std::ofstream out(argv[2]);
  const double R = 0.8, etaMax = 2.5, pTjetMin = 800., pTjetMax = 820., pTjetMinHadron = 600., dRMatchMax = 0.4;
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, R);

  long nRead = 0, nAccepted = 0, nNoFragVertex = 0;
  while (true) {
    GenEvent ev(Units::GEV, Units::MM);
    reader.read_event(ev);
    if (reader.failed()) break;
    ++nRead;

    std::vector<fastjet::PseudoJet> partons, hadrons;
    std::set<int> seen;
    bool foundFrag = false;
    for (const auto& v : ev.vertices()) {
      const auto& ins = v->particles_in();
      if (ins.empty()) continue;
      bool allPartons = true, hadronOut = false;
      for (const auto& p : ins) allPartons &= isParton(p->pid());
      for (const auto& p : v->particles_out()) hadronOut |= !isParton(p->pid()) && p->pid() != 22;
      if (!allPartons || !hadronOut) continue;
      foundFrag = true;
      for (const auto& p : ins) {
        if (!seen.insert(p->id()).second) continue;
        if (std::abs(p->momentum().eta()) > etaMax) continue;
        partons.push_back(toPJ(p));
      }
    }
    if (!foundFrag) { ++nNoFragVertex; continue; }

    for (const auto& p : ev.particles()) {
      if (p->status() != 1) continue;
      int a = std::abs(p->pid());
      if (a == 12 || a == 14 || a == 16) continue;
      if (p->momentum().pt() < 0.1 || std::abs(p->momentum().eta()) > etaMax) continue;
      if (isPromptPhoton(p)) continue;
      hadrons.push_back(toPJ(p));
    }

    fastjet::ClusterSequence csP(partons, jetDef), csH(hadrons, jetDef);
    auto pJets = fastjet::sorted_by_pt(csP.inclusive_jets(pTjetMin));
    if (pJets.empty() || pJets[0].pt() > pTjetMax) continue;
    auto hJets = fastjet::sorted_by_pt(csH.inclusive_jets(pTjetMinHadron));
    int iHad = -1;
    double best = dRMatchMax;
    for (size_t j = 0; j < hJets.size(); ++j) {
      double dR = hJets[j].delta_R(pJets[0]);
      if (dR < best) { best = dR; iHad = (int)j; }
    }
    if (iHad < 0) continue;

    auto hc = hJets[iHad].constituents(), pc = pJets[0].constituents();
    out << "E " << hc.size() << " " << pc.size() << "\n";
    char line[96];
    for (const auto* list : {&hc, &pc})
      for (const auto& c : *list) {
        std::snprintf(line, sizeof line, "%.6g %.6g %.6g\n", c.pt(), c.eta(), c.phi_std());
        out << line;
      }
    ++nAccepted;
  }
  std::cerr << "sherpa_jets: read " << nRead << " events, accepted " << nAccepted
            << ", without a hadronization vertex " << nNoFragVertex << "\n";
  return 0;
}
