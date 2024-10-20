// test_config_manager.cpp
#include <gtest/gtest.h>
#include "config_manager.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

class ConfigManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_config_file = "test_config.yml";
        createTestConfigFile();
    }

    void TearDown() override {
        fs::remove(test_config_file);
    }

    void createTestConfigFile() {
        std::ofstream config_file(test_config_file);
        config_file << "model_file1: data/training/html\n"
                    << "model_file2: data/training/email\n"
                    << "num_threads: 4\n"
                    << "pruning_threshold: 1e-5\n"
                    << "verbose_mode: true\n"
                    << "use_gpu: false\n";
        config_file.close();
    }

    std::string test_config_file;
};

TEST_F(ConfigManagerTest, TestLoadConfig) {
    ConfigManager config_manager(test_config_file);
    
    EXPECT_EQ(config_manager.getModelFile1(), "data/training/html");
    EXPECT_EQ(config_manager.getModelFile2(), "data/training/email");
    EXPECT_EQ(config_manager.getNumThreads(), 4);
    EXPECT_DOUBLE_EQ(config_manager.getPruningThreshold(), 1e-5);
    EXPECT_TRUE(config_manager.getVerboseMode());
    EXPECT_FALSE(config_manager.getUseGPU());
}

TEST_F(ConfigManagerTest, TestMissingFile) {
    EXPECT_THROW(ConfigManager("non_existent_file.yml"), std::runtime_error);
}

TEST_F(ConfigManagerTest, TestInvalidConfig) {
    std::ofstream config_file("invalid_config.yml");
    config_file << "invalid_yaml: :\n";
    config_file.close();

    EXPECT_THROW(ConfigManager("invalid_config.yml"), std::runtime_error);
    fs::remove("invalid_config.yml");
}