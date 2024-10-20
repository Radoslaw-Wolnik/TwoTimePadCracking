#include <gtest/gtest.h>
#include "language_model.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class LanguageModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data paths
        std::string test_data_dir = "data/testing";
        html_test_file = test_data_dir + "/html/sample_html_001.html";
        email_test_file = test_data_dir + "/email/sample_email_001.txt";
        plaintext_test_file = test_data_dir + "/plaintext/sample_plaintext_001.txt";
        
        // Load test data
        html_content = readFile(html_test_file);
        email_content = readFile(email_test_file);
        plaintext_content = readFile(plaintext_test_file);
        
        // Create models with training data
        html_model = std::make_unique<LanguageModel>("data/training/html");
        email_model = std::make_unique<LanguageModel>("data/training/email");
        plaintext_model = std::make_unique<LanguageModel>("data/training/plaintext");
    }
    
    std::string readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    
    std::string html_content, email_content, plaintext_content;
    std::string html_test_file, email_test_file, plaintext_test_file;
    std::unique_ptr<LanguageModel> html_model, email_model, plaintext_model;
};

TEST_F(LanguageModelTest, TestHtmlProbability) {
    double probability = html_model->getProbability(html_content);
    EXPECT_GT(probability, 0.0);
    EXPECT_LT(probability, 1.0);
}

TEST_F(LanguageModelTest, TestEmailProbability) {
    double probability = email_model->getProbability(email_content);
    EXPECT_GT(probability, 0.0);
    EXPECT_LT(probability, 1.0);
}

TEST_F(LanguageModelTest, TestPlaintextProbability) {
    double probability = plaintext_model->getProbability(plaintext_content);
    EXPECT_GT(probability, 0.0);
    EXPECT_LT(probability, 1.0);
}

TEST_F(LanguageModelTest, TestCrossModelProbabilities) {
    double html_prob_for_html = html_model->getProbability(html_content);
    double html_prob_for_email = html_model->getProbability(email_content);
    EXPECT_GT(html_prob_for_html, html_prob_for_email);
    
    double email_prob_for_email = email_model->getProbability(email_content);
    double email_prob_for_plaintext = email_model->getProbability(plaintext_content);
    EXPECT_GT(email_prob_for_email, email_prob_for_plaintext);
}

TEST_F(LanguageModelTest, TestPerplexity) {
    double html_perplexity = html_model->getPerplexity(html_content);
    EXPECT_GT(html_perplexity, 1.0);
    
    double email_perplexity = email_model->getPerplexity(email_content);
    EXPECT_GT(email_perplexity, 1.0);
    
    double plaintext_perplexity = plaintext_model->getPerplexity(plaintext_content);
    EXPECT_GT(plaintext_perplexity, 1.0);
}

TEST_F(LanguageModelTest, TestUnseenNGrams) {
    std::string unseen_content = "xyzxyzxyz";
    double probability = html_model->getProbability(unseen_content);
    EXPECT_GT(probability, 0.0);
    EXPECT_LT(probability, 1.0);
}