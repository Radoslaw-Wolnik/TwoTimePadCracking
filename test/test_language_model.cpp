// test_language_model.cpp
#include <gtest/gtest.h>
#include "language_model.h"

class LanguageModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        test_corpus = "test_corpus.txt";
        std::ofstream file(test_corpus);
        file << "This is a test corpus for language model testing.";
        file.close();

        model = std::make_unique<LanguageModel>(test_corpus);
    }

    void TearDown() override {
        std::remove(test_corpus.c_str());
    }

    std::string test_corpus;
    std::unique_ptr<LanguageModel> model;
};

TEST_F(LanguageModelTest, GetProbability) {
    EXPECT_GT(model->getProbability("This", ' '), 0.0);
    EXPECT_GT(model->getProbability("is", ' '), 0.0);
    EXPECT_EQ(model->getProbability("xyz", 'q'), 1.0 / 256);  // Uniform distribution for unknown context
}

TEST_F(LanguageModelTest, Perplexity) {
    double perplexity = model->getPerplexity("This is a test.");
    EXPECT_GT(perplexity, 0.0);
    EXPECT_LT(perplexity, 100.0);  // Arbitrary upper bound, adjust based on your model
}