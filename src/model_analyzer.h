// model_analyzer.h
#pragma once

#include "language_model.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

class ModelAnalyzer {
public:
    ModelAnalyzer(const LanguageModel& model);
    void showTopNGrams(int n, int top_k);
    void showCharacterDistribution();
    double calculatePerplexity(const std::string& text);
    void compareModels(const ModelAnalyzer& other);

private:
    const LanguageModel& model_;
    std::map<std::string, double> getNGramProbabilities(int n) const;
    void printDistribution(const std::map<char, double>& distribution);
};