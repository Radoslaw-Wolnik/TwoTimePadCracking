// test_decryptor.cpp
#include <gtest/gtest.h>
#include "decryptor.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class DecryptorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data paths
        std::string test_data_dir = "data/testing";
        html_test_file = test_data_dir + "/html/sample_html_001.html";
        email_test_file = test_data_dir + "/email/sample_email_001.txt";
        
        // Load test data
        html_content = readFile(html_test_file);
        email_content = readFile(email_test_file);
        
        // Create decryptor with training data
        decryptor = std::make_unique<Decryptor>("data/training/html", "data/training/email");
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
    
    std::string html_content, email_content;
    std::string html_test_file, email_test_file;
    std::unique_ptr<Decryptor> decryptor;
};

TEST_F(DecryptorTest, TestDecryptHtmlEmail) {
    std::string xor_text = xorStrings(html_content, email_content);
    auto result = decryptor->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    // Check if the decrypted texts are close to the original
    double html_similarity = calculateSimilarity(result.first, html_content);
    double email_similarity = calculateSimilarity(result.second, email_content);
    
    EXPECT_GT(html_similarity, 0.8);
    EXPECT_GT(email_similarity, 0.8);
}

TEST_F(DecryptorTest, TestDecryptHtmlHtml) {
    std::string html_content2 = readFile(test_data_dir + "/html/sample_html_002.html");
    std::string xor_text = xorStrings(html_content, html_content2);
    auto result = decryptor->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double similarity1 = calculateSimilarity(result.first, html_content);
    double similarity2 = calculateSimilarity(result.second, html_content2);
    
    EXPECT_GT(similarity1, 0.8);
    EXPECT_GT(similarity2, 0.8);
}

TEST_F(DecryptorTest, TestDecryptEmailEmail) {
    std::string email_content2 = readFile(test_data_dir + "/email/sample_email_002.txt");
    std::string xor_text = xorStrings(email_content, email_content2);
    auto result = decryptor->decrypt(xor_text);
    
    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    double similarity1 = calculateSimilarity(result.first, email_content);
    double similarity2 = calculateSimilarity(result.second, email_content2);
    
    EXPECT_GT(similarity1, 0.8);
    EXPECT_GT(similarity2, 0.8);
}

// Helper function to calculate similarity (you may need to implement this)
double calculateSimilarity(const std::string& s1, const std::string& s2) {
    // Implement a similarity measure, e.g., Levenshtein distance or n-gram similarity
    // For simplicity, let's use a character-based similarity
    int matching_chars = 0;
    for (size_t i = 0; i < std::min(s1.length(), s2.length()); ++i) {
        if (s1[i] == s2[i]) {
            matching_chars++;
        }
    }
    return static_cast<double>(matching_chars) / std::max(s1.length(), s2.length());
}