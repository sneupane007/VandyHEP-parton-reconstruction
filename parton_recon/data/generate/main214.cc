// Compare "parton-level" and "hadron-level" properties.
#include "Pythia8/Pythia.h"
#include "./json.hpp"
#include <map>
#include <set>
#include <vector>
#include <iostream>
using json = nlohmann::json;
json createGraph(const std::vector<std::vector<float>>& parton_node_features,
                 const std::vector<std::vector<int>>& parton_edge_index,
                 const std::vector<std::vector<float>>& parton_edge_features) {
    json graph;
    graph["node_features"] = parton_node_features;
    graph["edge_index"] = parton_edge_index;
    graph["edge_features"] = parton_edge_features;
    return graph;
}
void appendGraphToFile(const std::string& file_path, const json& graph, bool first_graph = false) {
    std::ofstream file;
    if (first_graph) {
        file.open(file_path, std::ofstream::out | std::ofstream::trunc);
        file << "[\n";
    } else {
        file.open(file_path, std::ofstream::out | std::ofstream::app);
        file << ",\n";
    }
    file << graph.dump(2);
    file.close();
}
void closeJsonArray(const std::string& file_path) {
    std::ofstream file;
    file.open(file_path, std::ofstream::out | std::ofstream::app);
    file << "\n]";
    file.close();
}
using namespace Pythia8;
// Generic routine to extract the particles that existed right before
// the hadronization machinery was invoked.
void getPartonLevelEvent(Event& event, Event& partonLevelEvent) {
  // Copy over all particles that existed right before hadronization.
  partonLevelEvent.reset();
  for (int i = 0; i < event.size(); ++i)
  if (event[i].isFinalPartonLevel()) {
    //  if (abs(event[i].eta()) > 2.5) continue;
    int iNew = partonLevelEvent.append( event[i] );
    // Set copied properties more appropriately: positive status,
    // original location as "mother", and with no daughters.
    partonLevelEvent[iNew].statusPos();
    partonLevelEvent[iNew].mothers( i, i);
    partonLevelEvent[iNew].daughters( 0, 0);
  }
}
//==========================================================================
// Generic routine to extract the particles that exist after the
// hadronization machinery. Normally not needed, since SlowJet
// contains the standard possibilities preprogrammed, but this
// method illustrates further discrimination.
void getHadronLevelEvent( Event& event, Event& hadronLevelEvent) {
  // Iterate over all final particles.
  hadronLevelEvent.reset();
  for (int i = 0; i < event.size(); ++i) {
    bool accept = false;
    if (event[i].isFinal()) accept = true;
    // Reject neutrinos.
    int idAbs = event[i].idAbs();
    if (idAbs == 12 || idAbs == 14 || idAbs == 16) accept = false;
    // Reject particles with pT < 0.1 GeV.
    if (event[i].pT() < 0.1) accept = false;
    // if (abs(event[i].eta()) > 2.5) accept = false;
    // Copy over accepted particles, with original location as "mother".
    if (accept) {
      int iNew = hadronLevelEvent.append( event[i] );
      hadronLevelEvent[iNew].mothers( i, i);
    }
  }
}
//==========================================================================
int main() {
  // Number of events, generated and listed ones.
  int nEvent    = 100;
  // Initialize graph objects.
//   json parton_graphs_json = json::array();
//   json hadron_graphs_json = json::array();
  // Generator. LHC process and output selection. Initialization.
  Pythia pythia;
  pythia.readString("Beams:eCM = 14000.");
  pythia.readString("PromptPhoton:qg2qgamma = on");
  pythia.readString("PhaseSpace:pTHatMin = 700.");
  pythia.readString("PartonShowers:model = 1"); // - check with model 1 as default and test 2 and 3
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return 1;
  // Parton and Hadron Level event records. Remeber to initalize.
  Event partonLevelEvent;
  partonLevelEvent.init("Parton Level event record", &pythia.particleData);
  Event hadronLevelEvent;
  hadronLevelEvent.init("Hadron Level event record", &pythia.particleData);
  //  Parameters for the jet finders. Need select = 1 to catch partons.
  int power = -1;       // -1 = anti-kT; 0 = C/A; 1 = kT.
  double radius = 0.8;       // Jet radius.
  double pTjetMin = 800.0; // Min jet pT.
  double pTjetMax = 820.0; // Max jet pT.
  double etaMax = 2.5;  // Pseudorapidity range of detector.
  int select = 1;       // Include all visibile particles (e.g. exclude neutrinos)
  // Set up anti-kT clustering, comparing parton and hadron levels.
  SlowJet antiKTpartons(power, radius, pTjetMin, etaMax, select);
  SlowJet antiKThadrons(power, radius, pTjetMin, etaMax, select);
  bool firstHadronGraph = true;
  bool firstPartonGraph = true;
  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    std::cout << "Event " << iEvent + 1 << " out of " << nEvent << endl;
    if (!pythia.next()) continue;
    std::vector<std::vector<float>> parton_node_features;
    std::vector<std::vector<int>> parton_edge_index;
    std::vector<std::vector<float>> parton_edge_features;
    std::vector<std::vector<float>> hadron_node_features;
    std::vector<std::vector<int>> hadron_edge_index;
    std::vector<std::vector<float>> hadron_edge_features;
    // Construct parton and hadron level event.
    getPartonLevelEvent(pythia.event, partonLevelEvent);
    getHadronLevelEvent(pythia.event, hadronLevelEvent);
    // Analyze jet properties and list first few analyses.
    antiKTpartons.analyze(partonLevelEvent);
    antiKThadrons.analyze(hadronLevelEvent);
    if(antiKTpartons.pT(0) > pTjetMax)
    continue;
    if(antiKTpartons.pT(0) < pTjetMin)
    continue;

   

    std::cout << "Leading parton jet [pT,eta,phi]: [" << antiKTpartons.pT(0) <<", " << antiKTpartons.y(0) << ", " << antiKTpartons.phi(0) << "]" << std::endl;
    for (auto i : antiKTpartons.constituents(0)) {
      std::vector<float> features;
      // Add particle properties to the node feature matrix
      // features.push_back(partonLevelEvent[i].m());
      features.push_back(partonLevelEvent[i].pT());
      // features.push_back(partonLevelEvent[i].e());
      features.push_back(partonLevelEvent[i].eta());
      features.push_back(partonLevelEvent[i].phi());
      //std::cout << "pT: " << partonLevelEvent[i].pT() << std::endl;
      //std::cout << "phi: " << partonLevelEvent[i].phi() << std::endl;
      //std::cout << "eta: " << partonLevelEvent[i].eta() << std::endl;
      parton_node_features.push_back(features);
      // Fastjet returns pythia event indices; convert them to 1, 2, ... , n
      map<int, int> transform;
      for (size_t idx = 0; idx < antiKTpartons.constituents(0).size(); ++idx) {
        transform[antiKTpartons.constituents(0)[idx]] = idx;
      }
            for (auto j : antiKTpartons.constituents(0)) {
                // All nodes i are connected to all other nodes j != i
                if (i == j)
                    continue;
                std::vector<int> edge = {transform[i], transform[j]};
                parton_edge_index.push_back(edge);
                // Use dR seperation between the particles as the edge feature
                double dEta = partonLevelEvent[i].eta() - partonLevelEvent[j].eta();
                double dPhi = abs(partonLevelEvent[i].phi() - partonLevelEvent[j].phi());
                if (dPhi > M_PI)
                    dPhi = 2. * M_PI - dPhi;
                float dR = sqrt(pow2(dEta) + pow2(dPhi));
                std::vector<float> edge_feature;
                edge_feature.push_back(dR);
                parton_edge_features.push_back(edge_feature);
            }
            //std::cout << " \n\n\n parton edge features size: " << parton_edge_features.size() << std::endl;
            //std::cout << " \n\n\n parton node features size: " << parton_node_features.size() << std::endl;
        }
    json parton_graph = createGraph(parton_node_features, parton_edge_index, parton_edge_features);
    appendGraphToFile("model1_parton_level_graphs_.json", parton_graph, firstPartonGraph);
    firstPartonGraph = false;
    // parton_graphs_json.push_back(parton_graph);
    
    // if(antiKThadrons.pT(0) > pTjetMax)
    // continue;
    // if(antiKThadrons.pT(0) < pTjetMin)
    // continue;

    // if(antiKThadrons.pT(0) > pTjetMax)
    // continue;
    // if(antiKThadrons.pT(0) < pTjetMin)
    // continue;
    std::cout << "Leading hadron jet [pT,eta,phi]: [" << antiKThadrons.pT(0) <<", " << antiKThadrons.y(0) << ", " << antiKThadrons.phi(0) << "]" << std::endl;

    for (auto i : antiKThadrons.constituents(0)) {
            std::vector<float> features;
            // Add particle properties to the node feature matrix
            // features.push_back(hadronLevelEvent[i].m());
            // features.push_back(hadronLevelEvent[i].px());
            // features.push_back(hadronLevelEvent[i].py());
            // features.push_back(hadronLevelEvent[i].pz());
            // features.push_back(hadronLevelEvent[i].e());
            features.push_back(hadronLevelEvent[i].pT());
            features.push_back(hadronLevelEvent[i].eta());
            features.push_back(hadronLevelEvent[i].phi());
            hadron_node_features.push_back(features);
            // Fastjet returns pythia event indices; convert them to 1, 2, ... , n
            map<int, int> transform;
            for (size_t idx = 0; idx < antiKThadrons.constituents(0).size(); ++idx) {
              transform[antiKThadrons.constituents(0)[idx]] = idx;
            }
            for (auto j : antiKThadrons.constituents(0)) {
                // All nodes i are connected to all other nodes j != i
                if (i == j)
                    continue;
                std::vector<int> edge = {transform[i], transform[j]};
                hadron_edge_index.push_back(edge);
                // Use dR seperation between the particles as the edge feature
                double dEta = hadronLevelEvent[i].eta() - hadronLevelEvent[j].eta();
                double dPhi = abs(hadronLevelEvent[i].phi() - hadronLevelEvent[j].phi());
                if (dPhi > M_PI)
                    dPhi = 2. * M_PI - dPhi;
                float dR = sqrt(pow2(dEta) + pow2(dPhi));
                std::vector<float> edge_feature;
                edge_feature.push_back(dR);
                hadron_edge_features.push_back(edge_feature);
            }
        }
        json hadron_graph = createGraph(hadron_node_features, hadron_edge_index, hadron_edge_features);
        appendGraphToFile("model1_hadron_level_graphs_.json", hadron_graph, firstHadronGraph);
        firstHadronGraph = false;
        // hadron_graphs_json.push_back(hadron_graph);
  // End of event loop.
  }
  closeJsonArray("model1_parton_level_graphs_.json");
  closeJsonArray("model1_hadron_level_graphs_.json");
  // closeJsonArray("default_parton_level_graphs.json");
  // closeJsonArray("default_hadron_level_graphs.json");
    // std::ofstream parton_file("parton_level_graphs.json");
    // parton_file << parton_graphs_json.dump(-1);
    // std::ofstream hadron_file("hadron_level_graphs.json");
    // hadron_file << hadron_graphs_json.dump(-1);
  // Done.
  return 0;
}