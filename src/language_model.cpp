// language_model.cpp
#include "language_model.h"
#include <algorithm>
#include <numeric>
#include <iostream>

LanguageModel::LanguageModel(const std::string& corpus_file, int n_gram_size)
    : n_gram_size_(n_gram_size) {
    buildModel(corpus_file);
    smoothProbabilities();
}

void LanguageModel::buildModel(const std::string& corpus_file) {
    std::ifstream file(corpus_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open corpus file: " + corpus_file);
    }

    std::unordered_map<std::string, std::vector<int>> count_table;
    std::string buffer(n_gram_size_ - 1, ' ');
    char c;
    while (file.get(c)) {
        buffer.push_back(c);
        if (buffer.size() >= n_gram_size_) {
            std::string context = buffer.substr(0, n_gram_size_ - 1);
            char next_char = buffer.back();
            if (count_table.find(context) == count_table.end()) {
                count_table[context] = std::vector<int>(CHAR_SET_SIZE, 0);
            }
            count_table[context][static_cast<unsigned char>(next_char)]++;
            buffer.erase(buffer.begin());
        }
    }

    // Convert counts to probabilities
    for (const auto& [context, counts] : count_table) {
        int total = std::accumulate(counts.begin(), counts.end(), 0);
        probability_table_[context] = std::vector<double>(CHAR_SET_SIZE);
        for (int i = 0; i < CHAR_SET_SIZE; i++) {
            probability_table_[context][i] = static_cast<double>(counts[i]) / total;
        }
    }
}

void LanguageModel::smoothProbabilities() {
    // Implement Witten-Bell smoothing here
    const double lambda = 0.5;  // Smoothing parameter
    
    for (auto& [context, probs] : probability_table_) {
        std::string backoff_context = context.substr(1);
        auto backoff_it = probability_table_.find(backoff_context);
        
        if (backoff_it != probability_table_.end()) {
            for (int i = 0; i < CHAR_SET_SIZE; i++) {
                probs[i] = lambda * probs[i] + (1 - lambda) * backoff_it->second[i];
            }
        }
    }
}

double LanguageModel::getProbability(const std::string& context, char next_char) const {
    auto it = probability_table_.find(context);
    if (it != probability_table_.end()) {
        return it->second[static_cast<unsigned char>(next_char)];
    } else if (context.length() > 1) {
        // Backoff to shorter context
        return getProbability(context.substr(1), next_char);
    } else {
        // Uniform distribution if no context found
        return 1.0 / CHAR_SET_SIZE;
    }
}

uint64_t LanguageModel::getBackoffContext(uint64_t context) const {
    // Implement context backing off for efficient lookup
    return context >> 8;  // Remove the oldest character from the context
}

void LanguageModel::saveModel(const std::string& file_path) const {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + file_path);
    }

    // Write n_gram_size
    file.write(reinterpret_cast<const char*>(&n_gram_size_), sizeof(n_gram_size_));

    // Write probability_table size
    size_t table_size = probability_table_.size();
    file.write(reinterpret_cast<const char*>(&table_size), sizeof(table_size));

    // Write probability_table entries
    for (const auto& [context, probs] : probability_table_) {
        size_t context_size = context.size();
        file.write(reinterpret_cast<const char*>(&context_size), sizeof(context_size));
        file.write(context.c_str(), context_size);
        file.write(reinterpret_cast<const char*>(probs.data()), sizeof(double) * CHAR_SET_SIZE);
    }
}

std::unique_ptr<LanguageModel> LanguageModel::loadModel(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for reading: " + file_path);
    }

    auto model = std::make_unique<LanguageModel>("", 0);  // Create empty model

    // Read n_gram_size
    file.read(reinterpret_cast<char*>(&model->n_gram_size_), sizeof(model->n_gram_size_));

    // Read probability_table size
    size_t table_size;
    file.read(reinterpret_cast<char*>(&table_size), sizeof(table_size));

    // Read probability_table entries
    for (size_t i = 0; i < table_size; ++i) {
        size_t context_size;
        file.read(reinterpret_cast<char*>(&context_size), sizeof(context_size));

        std::string context(context_size, '\0');
        file.read(&context[0], context_size);

        std::vector<double> probs(CHAR_SET_SIZE);
        file.read(reinterpret_cast<char*>(probs.data()), sizeof(double) * CHAR_SET_SIZE);

        model->probability_table_[context] = std::move(probs);
    }

    return model;
}

double LanguageModel::getPerplexity(const std::string& text) const {
    double log_prob = 0.0;
    int n = 0;

    std::string context(n_gram_size_ - 1, ' ');
    for (char c : text) {
        log_prob += std::log2(getProbability(context, c));
        context = context.substr(1) + c;
        ++n;
    }

    return std::pow(2, -log_prob / n);
}
