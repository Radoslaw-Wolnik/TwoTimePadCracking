// test_decryptor.cpp
#include <gtest/gtest.h>
#include "decryptor.h"

class DecryptorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test model files
        std::ofstream model1_file("test_model1.txt");
        model1_file << "Test model 1 content";
        model1_file.close();

        std::ofstream model2_file("test_model2.txt");
        model2_file << "Test model 2 content";
        model2_file.close();

        decryptor = std::make_unique<Decryptor>("test_model1.txt", "test_model2.txt");
    }

    void TearDown() override {
        std::remove("test_model1.txt");
        std::remove("test_model2.txt");
    }

    std::unique_ptr<Decryptor> decryptor;
};

TEST_F(DecryptorTest, DecryptBasic) {
    std::string xor_text = "Basic XOR text for testing";
    auto result = decryptor->decrypt(xor_text);

    EXPECT_EQ(result.first.length(), xor_text.length());
    EXPECT_EQ(result.second.length(), xor_text.length());
}