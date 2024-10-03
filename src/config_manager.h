// config_manager.h
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

class ConfigManager {
public:
    ConfigManager(const std::string& config_file);

    std::string getModelFile1() const;
    std::string getModelFile2() const;
    int getNumThreads() const;
    double getPruningThreshold() const;
    bool getVerboseMode() const;
    bool getUseGPU() const;

private:
    YAML::Node config_;
};