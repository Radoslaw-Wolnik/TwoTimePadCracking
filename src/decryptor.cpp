// decryptor.cpp
#include "decryptor.h"
#include "logger.h"
#include <iostream>

Decryptor::Decryptor(const std::string& model1_file, const std::string& model2_file) {
    Logger::get()->info("Loading language models...");
    model1_ = std::make_unique<LanguageModel>(model1_file);
    model2_ = std::make_unique<LanguageModel>(model2_file);
    Logger::get()->info("Language models loaded.");
    
    search_ = std::make_unique<ViterbiSearch>(*model1_, *model2_);
    visualizer_ = std::make_unique<Visualizer>();
    analyzer1_ = std::make_unique<ModelAnalyzer>(*model1_);
    analyzer2_ = std::make_unique<ModelAnalyzer>(*model2_);
}

void Decryptor::enableViterbiVisualization(const std::string& output_file) {
    viterbi_visualizer_ = std::make_shared<ViterbiVisualizer>();
    search_->setVisualizer(viterbi_visualizer_);
    viterbi_visualization_file_ = output_file;
}

std::pair<std::string, std::string> Decryptor::decrypt(const std::string& xor_text) {
    Logger::get()->info("Starting decryption process...");
    
    size_t total_steps = xor_text.length();
    std::vector<double> confidences;
    
    search_->setVerbose(verbose_);
    search_->setPruningThreshold(pruning_threshold_);
    search_->setNumThreads(num_threads_);
    search_->setUseGPU(use_gpu_);

    auto result = search_->decrypt(xor_text, 
        [this, &confidences, total_steps](size_t current, double confidence) {
            if (verbose_) {
                reportProgress(current, total_steps);
                confidences.push_back(confidence);
            }
        }
    );
    
    Logger::get()->info("Decryption complete.");
    
    if (verbose_) {
        visualizer_->showComparison(xor_text, result.first);
        visualizer_->showComparison(xor_text, result.second);
        visualizer_->showConfidenceMap(confidences);
        
        Logger::get()->info("Analyzing language models...");
        analyzer1_->showTopNGrams(3, 10);
        analyzer2_->showTopNGrams(3, 10);
        analyzer1_->showCharacterDistribution();
        analyzer2_->showCharacterDistribution();
        
        Logger::get()->info("Decrypted text 1: {}", result.first);
        Logger::get()->info("Decrypted text 2: {}", result.second);
    }
    
    if (viterbi_visualizer_) {
        viterbi_visualizer_->saveGraph(viterbi_visualization_file_);
        Logger::get()->info("Viterbi graph saved to {}", viterbi_visualization_file_);
    }
    
    return result;
}

void Decryptor::reportProgress(size_t current, size_t total) const {
    visualizer_->showProgress(static_cast<double>(current) / total);
}