// model_analyzer.cpp
#include "model_analyzer.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>

ModelAnalyzer::ModelAnalyzer(const LanguageModel& model) : model_(model) {}

void ModelAnalyzer::showTopNGrams(int n, int top_k) {
    auto ngrams = getNGramProbabilities(n);
    std::vector<std::pair<std::string, double>> ngram_vec(ngrams.begin(), ngrams.end());
    std::partial_sort(ngram_vec.begin(), ngram_vec.begin() + top_k, ngram_vec.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "Top " << top_k << " " << n << "-grams:\n";
    for (int i = 0; i < top_k && i < ngram_vec.size(); ++i) {
        std::cout << std::setw(10) << ngram_vec[i].first << ": " 
                  << std::fixed << std::setprecision(6) << ngram_vec[i].second << "\n";
    }
}

void ModelAnalyzer::showCharacterDistribution() {
    auto char_dist = getNGramProbabilities(1);
    std::map<char, double> distribution;
    for (const auto& [ch_str, prob] : char_dist) {
        distribution[ch_str[0]] = prob;
    }

    printDistribution(distribution);
}

double ModelAnalyzer::calculatePerplexity(const std::string& text) {
    return model_.getPerplexity(text);
}

void ModelAnalyzer::compareModels(const ModelAnalyzer& other) {
    auto this_dist = getNGramProbabilities(1);
    auto other_dist = other.getNGramProbabilities(1);

    double kl_divergence = 0.0;
    for (const auto& [ch, prob] : this_dist) {
        if (other_dist.count(ch) > 0) {
            kl_divergence += prob * std::log(prob / other_dist.at(ch));
        }
    }

    std::cout << "KL Divergence between models: " << kl_divergence << std::endl;
}

std::map<std::string, double> ModelAnalyzer::getNGramProbabilities(int n) const {
    std::map<std::string, double> ngrams;
    
    // Assuming model_ is a member variable of type LanguageModel
    const auto& probability_table = model_.getProbabilityTable();
    
    for (const auto& [context, probs] : probability_table) {
        if (context.length() == n - 1) {
            for (int i = 0; i < LanguageModel::CHAR_SET_SIZE; ++i) {
                std::string ngram = context + static_cast<char>(i);
                ngrams[ngram] = probs[i];
            }
        }
    }
    
    return ngrams;
}


void ModelAnalyzer::printDistribution(const std::map<char, double>& distribution) {
    std::cout << "Character distribution:\n";
    for (const auto& [ch, prob] : distribution) {
        std::cout << "'" << ch << "': " << std::string(static_cast<int>(prob * 100), '#') << "\n";
    }
}
