// test_visualizer.cpp
#include <gtest/gtest.h>
#include "visualizer.h"
#include <sstream>

class VisualizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        visualizer = std::make_unique<Visualizer>();
    }

    std::unique_ptr<Visualizer> visualizer;
};

TEST_F(VisualizerTest, TestShowProgress) {
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    visualizer->showProgress(0.5);

    std::cout.rdbuf(old);
    std::string output = buffer.str();

    EXPECT_TRUE(output.find("[") != std::string::npos);
    EXPECT_TRUE(output.find("]") != std::string::npos);
    EXPECT_TRUE(output.find("50.00%") != std::string::npos);
}

TEST_F(VisualizerTest, TestShowComparison) {
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    visualizer->showComparison("Hello", "Hallo");

    std::cout.rdbuf(old);
    std::string output = buffer.str();

    EXPECT_TRUE(output.find("Hello") != std::string::npos);
    EXPECT_TRUE(output.find("Hallo") != std::string::npos);
    EXPECT_TRUE(output.find("  ^") != std::string::npos);
}

TEST_F(VisualizerTest, TestShowConfidenceMap) {
    std::stringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

    std::vector<double> confidences = {0.1, 0.5, 0.9};
    visualizer->showConfidenceMap(confidences);

    std::cout.rdbuf(old);
    std::string output = buffer.str();

    EXPECT_TRUE(output.find("#         ") != std::string::npos);
    EXPECT_TRUE(output.find("#####     ") != std::string::npos);
    EXPECT_TRUE(output.find("#########") != std::string::npos);
}