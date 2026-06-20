#pragma once

#include <iostream>
#include <string>

namespace NeuroNet {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    NONE
};

class Logger {
public:
    static void SetLevel(LogLevel level) { current_level_ = level; }
    static LogLevel GetLevel() { return current_level_; }

    static void Debug(const std::string& message) { Log(LogLevel::DEBUG, message); }
    static void Info(const std::string& message) { Log(LogLevel::INFO, message); }
    static void Warning(const std::string& message) { Log(LogLevel::WARNING, message); }
    static void Error(const std::string& message) { Log(LogLevel::ERROR, message); }

private:
    static void Log(LogLevel level, const std::string& message) {
        if (level >= current_level_) {
            std::ostream& out = (level == LogLevel::ERROR) ? std::cerr : std::cout;
            out << "[" << LevelToString(level) << "] " << message << std::endl;
        }
    }

    static std::string LevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    static inline LogLevel current_level_ = LogLevel::INFO;
};

} // namespace NeuroNet
