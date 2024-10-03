// test_viterbi_search.cpp
#include <gtest/gtest.h>
#include "viterbi_search.h"

class ViterbiSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple language models for testing
        model1 = std::make_unique<LanguageModel>("model1.txt");
        model2 = std::make_unique<LanguageModel>("model2.txt");
        search = std::make_unique<ViterbiSearch>(*model1, *model2);
    }

    std::unique_ptr<LanguageModel> model1;
    std::unique_ptr<LanguageModel> model2;
    std::unique_ptr<ViterbiSearch> search;
};

TEST_F(ViterbiSearchTest, Decrypt) {
    std::string xor_text = "test xor text";
    auto result = search->decrypt(xor_text);

    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
    
    // XOR the results to verify they match the input
    std::string xor_result;
    for (size_t i = 0; i < xor_text.length(); ++i) {
        xor_result += static_cast<char>(result.first[i] ^ result.second[i]);
    }
    EXPECT_EQ(xor_result, xor_text);
}
