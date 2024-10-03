// viterbi_visualizer.cpp
#include "viterbi_visualizer.h"
#include <fstream>
#include <iostream>

ViterbiVisualizer::ViterbiVisualizer() {}

void ViterbiVisualizer::addState(int step, const std::string& state_id, double probability) {
    Vertex v = boost::add_vertex(graph_);
    graph_[v].step = step;
    graph_[v].state_id = state_id;
    graph_[v].probability = probability;
    vertex_map_[state_id] = v;
}

void ViterbiVisualizer::addTransition(const std::string& from_state, const std::string& to_state) {
    auto from_it = vertex_map_.find(from_state);
    auto to_it = vertex_map_.find(to_state);

    if (from_it != vertex_map_.end() && to_it != vertex_map_.end()) {
        boost::add_edge(from_it->second, to_it->second, graph_);
    }
}

void ViterbiVisualizer::saveGraph(const std::string& filename) {
    std::ofstream dot_file(filename);
    if (!dot_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    auto vertex_writer = [&](std::ostream& out, const Vertex& v) {
        out << "[label=\"" << graph_[v].state_id << "\\n"
            << "Step: " << graph_[v].step << "\\n"
            << "Prob: " << graph_[v].probability << "\"]";
    };

    boost::write_graphviz(dot_file, graph_, vertex_writer);
    std::cout << "Graph saved to " << filename << std::endl;
}