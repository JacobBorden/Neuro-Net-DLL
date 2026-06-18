#include "logger.h"

namespace NeuroNet {
    Logger::Logger() : current_level(LogLevel::INFO) {}

    Logger& Logger::GetInstance() {
        static Logger instance;
        return instance;
    }

    void Logger::SetLogLevel(LogLevel level) {
        current_level.store(level, std::memory_order_relaxed);
    }

    void Logger::Log(LogLevel level, const std::string& message) {
        if (level < current_level.load(std::memory_order_relaxed) || level == LogLevel::NONE) return;

        std::lock_guard<std::mutex> lock(log_mutex);
        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG: level_str = "[DEBUG] "; break;
            case LogLevel::INFO: level_str = "[INFO] "; break;
            case LogLevel::WARNING: level_str = "[WARNING] "; break;
            case LogLevel::ERROR: level_str = "[ERROR] "; break;
            default: level_str = "[UNKNOWN] "; break;
        }

        if (level == LogLevel::ERROR || level == LogLevel::WARNING) {
            std::cerr << level_str << message << std::endl;
        } else {
            std::cout << level_str << message << std::endl;
        }
    }
}
