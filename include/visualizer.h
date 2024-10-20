// Visualizer.h
#pragma once
#include <string>
#include <vector>

class Visualizer {
public:
    void showProgress(double progress);
    void showComparison(const std::string& original, const std::string& decrypted);
    void showConfidenceMap(const std::vector<double>& confidences);

private:
    static constexpr int BAR_WIDTH = 50;
    void clearLine() const;
};