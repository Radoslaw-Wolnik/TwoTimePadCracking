#include "decryptor.h"
#include "config_manager.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <cxxopts.hpp>

std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("two_time_pad_cracker", "A tool to crack two-time pad encryption");

    options.add_options()
        ("c,config", "Path to the configuration file", cxxopts::value<std::string>())
        ("x,xor", "Path to the XORed text file", cxxopts::value<std::string>())
        ("v,verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false"))
        ("l,log", "Log file path", cxxopts::value<std::string>()->default_value("two_time_pad_cracker.log"))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("config") || !result.count("xor")) {
        std::cerr << "Missing required arguments. Use --help for usage information." << std::endl;
        return 1;
    }

    try {
        Logger::init(result["log"].as<std::string>());
        Logger::get()->info("Two-Time Pad Cracker started");

        ConfigManager config(result["config"].as<std::string>());
        Decryptor decryptor(config.getModelFile1(), config.getModelFile2());
        decryptor.setVerbose(result["verbose"].as<bool>());
        decryptor.setPruningThreshold(config.getPruningThreshold());
        decryptor.setNumThreads(config.getMaxThreads());
        
        if (config.getUseGPU()) {
            decryptor.enableGPUAcceleration();
        }
        
        if (config.getEnableViterbiVisualization()) {
            decryptor.enableViterbiVisualization(config.getViterbiVisualizationFile());
        }

        std::string xor_text = readFile(result["xor"].as<std::string>());
        
        Logger::get()->info("XORed text length: {} bytes", xor_text.length());
        
        auto [plaintext1, plaintext2] = decryptor.decrypt(xor_text);
        
        std::string output_dir = config.getOutputDirectory();
        std::ofstream out1(output_dir + "/plaintext1.txt");
        std::ofstream out2(output_dir + "/plaintext2.txt");
        out1 << plaintext1;
        out2 << plaintext2;
        
        Logger::get()->info("Decryption completed successfully");
        Logger::get()->info("Results saved to {} and {}", output_dir + "/plaintext1.txt", output_dir + "/plaintext2.txt");

    } catch (const std::exception& e) {
        Logger::get()->error("An error occurred: {}", e.what());
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}