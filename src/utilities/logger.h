#pragma once
#include <iostream>
#include <mutex>
#include <string>
#include <atomic>

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
        static Logger& GetInstance();
        void SetLogLevel(LogLevel level);
        void Log(LogLevel level, const std::string& message);
    private:
        Logger();
        std::atomic<LogLevel> current_level;
        std::mutex log_mutex;
    };
}

#define LOG_DEBUG(msg) ::NeuroNet::Logger::GetInstance().Log(::NeuroNet::LogLevel::DEBUG, msg)
#define LOG_INFO(msg) ::NeuroNet::Logger::GetInstance().Log(::NeuroNet::LogLevel::INFO, msg)
#define LOG_WARNING(msg) ::NeuroNet::Logger::GetInstance().Log(::NeuroNet::LogLevel::WARNING, msg)
#define LOG_ERROR(msg) ::NeuroNet::Logger::GetInstance().Log(::NeuroNet::LogLevel::ERROR, msg)
