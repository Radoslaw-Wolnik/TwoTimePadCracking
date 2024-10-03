// logger.h
#pragma once
#include <spdlog/spdlog.h>
#include <memory>

class Logger {
public:
    static void init(const std::string& log_file);
    static std::shared_ptr<spdlog::logger> get();

private:
    static std::shared_ptr<spdlog::logger> logger_;
};