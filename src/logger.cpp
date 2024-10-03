// logger.cpp
#include "logger.h"
#include <spdlog/sinks/basic_file_sink.h>

std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

void Logger::init(const std::string& log_file) {
    logger_ = spdlog::basic_logger_mt("two_time_pad_cracker", log_file);
    spdlog::set_default_logger(logger_);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
}

std::shared_ptr<spdlog::logger> Logger::get() {
    if (!logger_) {
        throw std::runtime_error("Logger not initialized. Call Logger::init() first.");
    }
    return logger_;
}
