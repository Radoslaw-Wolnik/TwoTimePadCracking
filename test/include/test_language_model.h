#pragma once

#include <gtest/gtest.h>
#include "language_model.h"
#include <memory>
#include <string>

class LanguageModelTest : public ::testing::Test {
protected:
    void SetUp() override;
    std::string readFile(const std::string& filename);

    std::string html_content, email_content, plaintext_content;
    std::string html_test_file, email_test_file, plaintext_test_file;
    std::unique_ptr<LanguageModel> html_model, email_model, plaintext_model;
};