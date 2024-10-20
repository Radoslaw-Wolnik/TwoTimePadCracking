#pragma once

#include <gtest/gtest.h>
#include "config_manager.h"
#include <string>

class ConfigManagerTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    void createTestConfigFile();

    std::string test_config_file;
};