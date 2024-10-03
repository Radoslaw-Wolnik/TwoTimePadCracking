// Visualizer.cpp
#include "Visualizer.h"
#include <iostream>
#include <iomanip>

void Visualizer::showProgress(double progress) {
    clearLine();
    int pos = static_cast<int>(BAR_WIDTH * progress);
    std::cout << "[";
    for (int i = 0; i < BAR_WIDTH; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << (progress * 100.0) << "%\r";
    std::cout.flush();
}

void Visualizer::showComparison(const std::string& original, const std::string& decrypted) {
    std::cout << "\nComparison:\n";
    std::cout << "Original:  " << original << "\n";
    std::cout << "Decrypted: " << decrypted << "\n";
    std::cout << "Diff:      ";
    for (size_t i = 0; i < original.length() && i < decrypted.length(); ++i) {
        std::cout << (original[i] == decrypted[i] ? ' ' : '^');
    }
    std::cout << "\n";
}

void Visualizer::showConfidenceMap(const std::vector<double>& confidences) {
    std::cout << "\nConfidence Map:\n";
    for (double confidence : confidences) {
        int level = static_cast<int>(confidence * 10);
        std::cout << std::string(level, '#') << std::string(10 - level, ' ') << " ";
    }
    std::cout << "\n";
}

void Visualizer::clearLine() const {
    std::cout << "\r" << std::string(80, ' ') << "\r";
}