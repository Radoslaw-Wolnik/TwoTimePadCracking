// test_viterbi_search.cpp
#include <gtest/gtest.h>
#include "viterbi_search.h"
#include "language_model.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class ViterbiSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data paths
        std::string test_data_dir = "data/testing";
        html_test_file = test_data_dir + "/html/sample_html_001.html";
        email_test_file = test_data_dir + "/email/sample_email_001.txt";
        
        // Load test data
        html_content = readFile(html_test_file);
        email_content = readFile(email_test_file);
        
        // Create language models with training data
        html_model = std::make_unique<LanguageModel>("data/training/html");
        email_model = std::make_unique<LanguageModel>("data/training/email");
        
        // Create ViterbiSearch object
        viterbi_search = std::make_unique<ViterbiSearch>(*html_model, *email_model);
    }
    
    std::string readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    
    std::string xorStrings(const std::string& s1, const std::string& s2) {
        std::string result;
        for (size_t i = 0; i < std::min(s1.length(), s2.length()); ++i) {
            result += static_cast<char>(s1[i] ^ s2[i]);
        }
        return result;
    }
    
    double calculateSimilarity(const std::string& s1, const std::string& s2) {
        int matching_chars = 0;
        for (size_t i = 0; i < std::min(s1.length(), s2.length()); ++i) {
            if (s1[i] == s2[i]) {
                matching_chars++;
            }
        }
        return static_cast<double>(matching_chars) / std::max(s1.length(), s2.length());
    }
    
    std::string html_content, email_content;
    std::string html_test_file, email_test_file;
    std::unique_ptr<LanguageModel> html_model, email_model;
    std::unique_ptr<ViterbiSearch> viterbi_search;
};

TEST_F(ViterbiSearchTest, TestDecrypt) {
    std::string xor_text = xorStrings(html_content, email_content);
    auto result = viterbi_search->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double html_similarity = calculateSimilarity(result.first, html_content);
    double email_similarity = calculateSimilarity(result.second, email_content);
    
    EXPECT_GT(html_similarity, 0.8);
    EXPECT_GT(email_similarity, 0.8);
}

TEST_F(ViterbiSearchTest, TestPruning) {
    viterbi_search->setPruningThreshold(1e-5);
    std::string xor_text = xorStrings(html_content, email_content);
    auto result = viterbi_search->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double html_similarity = calculateSimilarity(result.first, html_content);
    double email_similarity = calculateSimilarity(result.second, email_content);
    
    EXPECT_GT(html_similarity, 0.7);  // Lowered expectation due to pruning
    EXPECT_GT(email_similarity, 0.7);
}

TEST_F(ViterbiSearchTest, TestParallelDecryption) {
    viterbi_search->setNumThreads(4);
    std::string xor_text = xorStrings(html_content, email_content);
    auto result = viterbi_search->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double html_similarity = calculateSimilarity(result.first, html_content);
    double email_similarity = calculateSimilarity(result.second, email_content);
    
    EXPECT_GT(html_similarity, 0.8);
    EXPECT_GT(email_similarity, 0.8);
}

TEST_F(ViterbiSearchTest, TestDifferentLengths) {
    std::string shortened_html = html_content.substr(0, html_content.length() / 2);
    std::string xor_text = xorStrings(shortened_html, email_content);
    auto result = viterbi_search->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double html_similarity = calculateSimilarity(result.first, shortened_html);
    double email_similarity = calculateSimilarity(result.second, email_content.substr(0, xor_text.length()));
    
    EXPECT_GT(html_similarity, 0.7);
    EXPECT_GT(email_similarity, 0.7);
}

TEST_F(ViterbiSearchTest, TestVerboseMode) {
    viterbi_search->setVerbose(true);
    std::string xor_text = xorStrings(html_content, email_content);
    
    testing::internal::CaptureStdout();
    auto result = viterbi_search->decrypt(xor_text);
    std::string output = testing::internal::GetCapturedStdout();
    
    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(output.find("ViterbiSearch") != std::string::npos);
}

TEST_F(ViterbiSearchTest, TestCancellation) {
    std::string long_html = std::string(1000000, 'a');  // Very long input
    std::string long_email = std::string(1000000, 'b');
    std::string xor_text = xorStrings(long_html, long_email);
    
    std::thread cancellation_thread([this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        viterbi_search->cancel();
    });
    
    EXPECT_THROW(viterbi_search->decrypt(xor_text), std::runtime_error);
    
    cancellation_thread.join();
}