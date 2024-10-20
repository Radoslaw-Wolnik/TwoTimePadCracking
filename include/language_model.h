// language_model.h
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <cmath>

class LanguageModel {
public:
    LanguageModel(const std::string& corpus_file, int n_gram_size = 7);

    double getProbability(const std::string& context, char next_char) const;
    
    // New methods
    void saveModel(const std::string& file_path) const;
    static std::unique_ptr<LanguageModel> loadModel(const std::string& file_path);
    double getPerplexity(const std::string& text) const;

private:
    static constexpr int CHAR_SET_SIZE = 256;  // Assuming 8-bit characters
    int n_gram_size_;
    std::unordered_map<std::string, std::vector<double>> probability_table_;
    
    void buildModel(const std::string& corpus_file);
    void smoothProbabilities();
    uint64_t getBackoffContext(uint64_t context) const;
};