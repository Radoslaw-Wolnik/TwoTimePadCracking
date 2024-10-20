#pragma once

#include <gtest/gtest.h>
#include "decryptor.h"
#include <memory>
#include <string>

class DecryptorTest : public ::testing::Test {
protected:
    void SetUp() override;
    std::string readFile(const std::string& filename);
    std::string xorStrings(const std::string& s1, const std::string& s2);
    double calculateSimilarity(const std::string& s1, const std::string& s2);

    std::string html_content, email_content;
    std::string html_test_file, email_test_file;
    std::unique_ptr<Decryptor> decryptor;
};