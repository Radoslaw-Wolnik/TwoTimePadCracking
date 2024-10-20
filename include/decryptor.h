// decryptor.h
#pragma once

#include "language_model.h"
#include "viterbi_search.h"
#include "visualizer.h"
#include "model_analyzer.h"
#include "viterbi_visualizer.h"
#include <memory>
#include <string>

class Decryptor {
public:
    Decryptor(const std::string& model1_file, const std::string& model2_file);
    std::pair<std::string, std::string> decrypt(const std::string& xor_text);
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setPruningThreshold(double threshold) { pruning_threshold_ = threshold; }
    void setNumThreads(int num_threads) { num_threads_ = num_threads; }
    void enableGPUAcceleration() { use_gpu_ = true; }
    void enableViterbiVisualization(const std::string& output_file);

private:
    std::unique_ptr<LanguageModel> model1_;
    std::unique_ptr<LanguageModel> model2_;
    std::unique_ptr<ViterbiSearch> search_;
    std::unique_ptr<Visualizer> visualizer_;
    std::unique_ptr<ModelAnalyzer> analyzer1_;
    std::unique_ptr<ModelAnalyzer> analyzer2_;
    std::shared_ptr<ViterbiVisualizer> viterbi_visualizer_;

    bool verbose_ = false;
    double pruning_threshold_ = 1e-5;
    int num_threads_ = 1;
    bool use_gpu_ = false;
    std::string viterbi_visualization_file_;

    void reportProgress(size_t current, size_t total) const;
};