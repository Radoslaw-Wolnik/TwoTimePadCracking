// config_manager.cpp
#include "config_manager.h"
#include <stdexcept>

ConfigManager::ConfigManager(const std::string& config_file) {
    try {
        config_ = YAML::LoadFile(config_file);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error loading config file: " + std::string(e.what()));
    }
}

std::string ConfigManager::getModelFile1() const {
    return config_["model_file1"].as<std::string>();
}

std::string ConfigManager::getModelFile2() const {
    return config_["model_file2"].as<std::string>();
}

int ConfigManager::getNumThreads() const {
    return config_["num_threads"].as<int>();
}

double ConfigManager::getPruningThreshold() const {
    return config_["pruning_threshold"].as<double>();
}

bool ConfigManager::getVerboseMode() const {
    return config_["verbose_mode"].as<bool>();
}

bool ConfigManager::getUseGPU() const {
    return config_["use_gpu"].as<bool>();
}