// viterbi_visualizer.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

class ViterbiVisualizer {
public:
    ViterbiVisualizer();
    void addState(int step, const std::string& state_id, double probability);
    void addTransition(const std::string& from_state, const std::string& to_state);
    void saveGraph(const std::string& filename);

private:
    struct VertexProperties {
        int step;
        std::string state_id;
        double probability;
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

    Graph graph_;
    std::map<std::string, Vertex> vertex_map_;
};