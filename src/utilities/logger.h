#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>

namespace NeuroNet {

class Logger {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        NONE
    };

    static Logger& GetInstance();

    void SetLevel(Level level);
    Level GetLevel() const;

    void SetOutputFile(const std::string& filename);
    void CloseOutputFile();
    void LogToConsole(bool enable);

    void Log(Level level, const std::string& message, const char* file = nullptr, int line = 0);

    // Helpers
    void Debug(const std::string& message, const char* file = nullptr, int line = 0);
    void Info(const std::string& message, const char* file = nullptr, int line = 0);
    void Warning(const std::string& message, const char* file = nullptr, int line = 0);
    void Error(const std::string& message, const char* file = nullptr, int line = 0);

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string LevelToString(Level level) const;

    Level m_Level;
    bool m_LogToConsole;
    std::unique_ptr<std::ofstream> m_FileStream;
    std::mutex m_Mutex;
};

// Macros for easy logging
#define LOG_DEBUG(msg) ::NeuroNet::Logger::GetInstance().Debug(msg, __FILE__, __LINE__)
#define LOG_INFO(msg)  ::NeuroNet::Logger::GetInstance().Info(msg, __FILE__, __LINE__)
#define LOG_WARN(msg)  ::NeuroNet::Logger::GetInstance().Warning(msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) ::NeuroNet::Logger::GetInstance().Error(msg, __FILE__, __LINE__)

} // namespace NeuroNet
